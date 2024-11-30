#include "main.hpp"
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "parser.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <emmintrin.h>
#include <immintrin.h>

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

// Exit flag
volatile bool g_stop = false;

// Guiding prefix for original lines of text
std::string iprefix = "JP: ";

// En: prefix doesn't contain a space, for an interesting reason.
// For Chinese models it may turn into an explicit space token.
// In English text such token usually doesn't appear.
// However, it may appear in Chinese texts as a delimiter.
// This may provoke Chinese-speaking model to output Chinese.
std::string isuffix = "En:";

std::string g_neg;

// Translation example injected after prompt
std::string example = "JP: この物語はフィクションです。\n"
					  "En: This story is a work of fiction.\n";

std::string cache_prefix;

// Print line, return original speaker name if first encountered
std::string print_line(line_id id, std::string* line = nullptr, bool stream = false)
{
	std::string out = apply_replaces(g_lines[id].text, false, 0);
	std::string speaker;
	if (g_lines[id].name.starts_with("選択肢#")) {
		// Insert translated choice selection prefix
		speaker += " Choice ";
		speaker += g_lines[id].name.substr("選択肢#"sv.size());
	} else if (!g_lines[id].name.empty()) {
		// Find registered speaker translation
		std::lock_guard lock(g_mutex);

		const auto& found = g_dict.at(g_lines[id].name);
		if (!found.first.empty()) {
			speaker += " ";
			speaker += found.first;
		}
	}

	if (g_mode == op_mode::print_only) {
		// Passthrough mode (print-only): add \ for multiline input
		std::cout << iprefix << g_lines[id].name << out << "\\" << std::endl;
		// Add Ctrl+D for same-line output
		std::cout << isuffix << speaker << "\04" << std::flush;
	}

	if (stream) {
		// Print colored output without prefixes
		std::cout << g_esc.orig << *line << g_lines[id].name << out << g_esc.reset << std::endl;
	}

	if (line) {
		line->append(iprefix);
		line->append(g_lines[id].name);
		line->append(out);
		line->append("\n");
	}

	return speaker;
}

std::u16string squeeze_line(const std::string& line)
{
	// For opportunistic compatibility with "bad" hooks which repeat characters, or remove repeats
	std::u16string r;
	std::string_view next;
	char16_t prev{};
	for (const char& c : line) {
		next = {};
		char32_t utf32 = 0;
		// Decode UTF-8 sequence
		if ((c & 0x80) == 0) {
			next = std::string_view(&c, 1);
			utf32 = c;
		} else if ((c & 0xe0) == 0xc0) {
			next = std::string_view(&c, 2);
			utf32 = (c & 0x1f);
		} else if ((c & 0xf0) == 0xe0) {
			next = std::string_view(&c, 3);
			utf32 = (c & 0x0f);
		} else if ((c & 0xf8) == 0xf0) {
			next = std::string_view(&c, 4);
			utf32 = (c & 0x07);
		}
		for (auto& ch : next) {
			// Check for null terminator (`next` could be out of bounds)
			if (ch == 0)
				return r;
			if (&ch > next.data() && (ch & 0xc0) != 0x80) {
				// Invalid UTF-8 sequence
				next = {};
				prev = 0;
				break;
			}
			if (&ch > next.data()) {
				utf32 <<= 6;
				utf32 |= (ch & 0x3f);
			}
		}
		// Saturate and normalize full-width to ASCII
		char16_t utf16 = std::min(utf32, U'\xffff');
		if (utf16 >= u'！' && utf16 <= u'～')
			utf16 -= u'！' - '!';

		if (!next.empty() && utf16 != prev) {
			// Skip spaces (both ASCII and full-width) and control characters
			if (next == "　"sv || next == " "sv)
				continue;
			if (next[0] + 0u < 32 || next[0] == '\x7f') {
				prev = 0;
				continue;
			}
			// Append non-repeating code
			r += utf16;
			prev = utf16;
		}
	}
	return r;
}

std::u32string to_utf32(const std::string& line)
{
	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
	return converter.from_bytes(line);
}

std::string to_utf8(const std::u32string& line)
{
	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
	return converter.to_bytes(line);
}

// Return the "cost" of changing strings s to t
// Implementation of https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
uint levenshtein_distance(std::u16string_view s, std::u16string_view t)
{
	if ((s.size() | t.size()) > 255)
		throw std::out_of_range("levenshtein_distance: strings too big");
	std::array<unsigned char, 256> v1_, v0_;
	auto v0 = v0_.data();
	auto v1 = v1_.data();
	for (uint i = 0; i < t.size() + 1; i++)
		v0[i] = i;
	for (uint i = 0; i < s.size(); i++) {
		v1[0] = i + 1;
		for (uint j = 0; j < t.size(); j++) {
			const uint del_cost = v0[j + 1] + 1;
			const uint ins_cost = v1[j] + 1;
			const uint sub_cost = v0[j] + (s[i] != t[j]);
			v1[j + 1] = std::min({del_cost, ins_cost, sub_cost});
		}
		std::swap(v0, v1);
	}
	return v0[t.size()];
}

// Stored "columns" of g_strings for linear memory access
static std::vector<std::vector<char16_t>> s_char_columns(255);

// Return vector of the most matching strings
std::vector<std::u16string_view> levenshtein_distance(uint skip, std::span<const std::u16string_view> span, std::u16string_view t, uint last = -1)
{
	std::vector<std::u16string_view> result;
	const auto pitch = (span.size() + 31) & -32;
	const auto flat_size = pitch * (t.size() + 1);
	static constexpr auto av = std::align_val_t{32};
	auto v0_flat = new (av) unsigned char[flat_size];
	auto v1_flat = new (av) unsigned char[flat_size];
	char16_t* ith = nullptr;
	for (uint i = 0; i < t.size() + 1; i++) {
		std::memset(v0_flat + i * pitch, i, pitch);
	}
	std::size_t max_str = span.empty() ? 0 : span[0].size();
	std::size_t min_str = 1;
	uint k0 = 0;
	uint k1 = span.size();
	for (uint i = 0; i < max_str; i++) {
		ith = s_char_columns[i].data() + skip;
		auto v0j = v0_flat;
		auto v0j2 = v0j + pitch;
		auto v1j = v1_flat;
		auto v1j2 = v1j + pitch;
#if defined(__SSE2__)
		for (uint k = k0 & -16; k < k1; k += 16) {
			_mm_store_si128((__m128i*)(v1j + k), _mm_set1_epi8(i + 1));
		}
#else
		std::memset(v1j + k0, i + 1, k1 - k0);
#endif
		for (uint j = 0; j < t.size(); j++) {
#if defined(__AVX2__)
			auto tj = _mm256_set1_epi16(t[j]);
			for (uint k = k0 & -31; k < k1; k += 32) {
				auto del_cost = _mm256_add_epi8(_mm256_load_si256((const __m256i*)(v0j2 + k)), _mm256_set1_epi8(1));
				auto ins_cost = _mm256_add_epi8(_mm256_load_si256((const __m256i*)(v1j + k)), _mm256_set1_epi8(1));
				auto m0 = _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i_u*)(ith + k)), tj);
				auto m1 = _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i_u*)(ith + k + 16)), tj);
				auto value = _mm256_add_epi8(_mm256_load_si256((const __m256i*)(v0j + k)), _mm256_set1_epi8(1));
				auto mask = _mm256_permute4x64_epi64(_mm256_packs_epi16(m0, m1), 0xd8);
				auto sub_cost = _mm256_add_epi8(value, mask);
				auto min_cost = _mm256_min_epu8(_mm256_min_epu8(del_cost, ins_cost), sub_cost);
				_mm256_store_si256((__m256i*)(v1j2 + k), min_cost);
			}
#elif defined(__SSE2__)
			auto tj = _mm_set1_epi16(t[j]);
			for (uint k = k0 & -16; k < k1; k += 16) {
				auto del_cost = _mm_add_epi8(_mm_load_si128((const __m128i*)(v0j2 + k)), _mm_set1_epi8(1));
				auto ins_cost = _mm_add_epi8(_mm_load_si128((const __m128i*)(v1j + k)), _mm_set1_epi8(1));
				auto m0 = _mm_cmpeq_epi16(_mm_loadu_si128((const __m128i_u*)(ith + k)), tj);
				auto m1 = _mm_cmpeq_epi16(_mm_loadu_si128((const __m128i_u*)(ith + k + 8)), tj);
				auto value = _mm_add_epi8(_mm_load_si128((const __m128i*)(v0j + k)), _mm_set1_epi8(1));
				auto mask = _mm_packs_epi16(m0, m1);
				auto sub_cost = _mm_add_epi8(value, mask);
				auto min_cost = _mm_min_epu8(_mm_min_epu8(del_cost, ins_cost), sub_cost);
				_mm_store_si128((__m128i*)(v1j2 + k), min_cost);
			}
// 			auto tj = _mm512_set1_epi16(t[j]);
// 			for (uint k = k0 & -63; k < k1; k += 64) {
// 				auto del_cost = _mm512_add_epi8(_mm512_load_epi8(v0j2 + k), _mm512_set1_epi8(1));
// 				auto ins_cost = _mm512_add_epi8(_mm512_load_epi8(v1j + k), _mm512_set1_epi8(1));
// 				__mmask64 mask = _mm512_cmp_epi16_mask(_mm512_loadu_epi16(ith + k), tj, _MM_CMPINT_NE);
// 				__mmask64 mask2 = _mm512_cmp_epi16_mask(_mm512_loadu_epi16(ith + k + 32), tj, _MM_CMPINT_NE);
// 				mask |= mask2 << 32;
// 				auto value = _mm512_load_epi8(v0j + k);
// 				auto sub_cost = _mm512_mask_add_epi8(value, mask, value, _mm512_set1_epi8(1));
// 				auto min_cost = _mm512_min_epu8(_mm512_min_epu8(del_cost, ins_cost), sub_cost);
// 				_mm512_store_epi8(v1j2 + k, min_cost);
// 			}
#else
			for (uint k = k0; k < k1; k++) {
				uint del_cost = v0j2[k] + 1;
				uint ins_cost = v1j[k] + 1;
				uint sub_cost = v0j[k] + (ith[k] != t[j]);
				v1j2[k] = std::min({del_cost, ins_cost, sub_cost});
			}
#endif

			v0j += pitch;
			v0j2 += pitch;
			v1j += pitch;
			v1j2 += pitch;
		}
		for (uint k = k0; k < k1; k++) {
			auto s = span[k];
			if (i + 1 == s.size()) {
				const uint res = v1j[k];
				//if (res != levenshtein_distance(s, t))
				//	throw std::runtime_error("Batch failed: " + std::to_string(res));
				if (res < last) {
					last = res;
					result.clear();
				}
				if (res == last) {
					result.push_back(s);
					if (res != levenshtein_distance(s, t))
						throw std::runtime_error("Bug: levenshtein_distance batch failed.");
				}
				// TODO: fix optimizations
				//max_str = std::min(max_str, last + s.size());
				// if (s.size() > last)
				// 	min_str = std::max(min_str, s.size() - last);
				//if (s.size() > max_str)
				//	k0 = std::max(k0, k + 1);
				// while (s.size() < min_str) {
				// 	k1 = k;
				// 	s = span[--k];
				// }
			}
		}
		std::swap(v0_flat, v1_flat);
	}
	operator delete[](v0_flat, av);
	operator delete[](v1_flat, av);
	return result;
}

namespace fs = std::filesystem;

static std::string names_path, replaces_path;

static fs::path vnsleuth_path;

void update_names(const std::string& path)
{
	if (path.empty())
		return;
	const auto path_tmp = path + ".tmp";
	std::ofstream names(path_tmp, std::ios_base::trunc);
	if (names.is_open()) {
		for (auto& [orig, pair] : g_dict) {
			if (pair.first != ":")
				names << orig << pair.first << pair.second << "\n";
			else
				names << orig << "\n";
		}
		names.close();
		fs::rename(path_tmp, path);
		// Workaround to make cache files always appear as last modified
		fs::last_write_time(path, fs::last_write_time(path) - 1s);
	} else {
		std::cerr << "Failed to open: " << path << ".tmp" << std::endl;
	}
}

bool update_segment(uint seg, bool upd_names = true, uint count = -1)
{
	std::lock_guard lock(g_mutex);

	if (upd_names) {
		update_names(names_path);
	}

	std::string dump = "SRC:";
	dump += g_lines.segs[seg].src_name;
	dump += '\n';
	for (auto& name : g_lines.segs[seg].prev_segs) {
		dump += "PREV:";
		dump += name;
		dump += '\n';
	}
	for (auto& line : g_lines.segs[seg].lines) {
		if (count == 0)
			break;
		if (line.tr_text.empty())
			break;
		dump += line.tr_text;
		count--;
	}
	if (count)
		dump += g_lines.segs[seg].tr_tail;
	auto tmp = vnsleuth_path / (g_lines.segs[seg].cache_name + ".tmp");
	std::ofstream file(tmp, std::ios_base::trunc | std::ios_base::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open " << g_lines.segs[seg].cache_name << ".tmp" << std::endl;
		return false;
	}
	file.write(dump.data(), dump.size());
	file.close();
	auto dst = vnsleuth_path / g_lines.segs[seg].cache_name;
	fs::rename(tmp, dst);
	if (!upd_names)
		fs::last_write_time(dst, fs::last_write_time(dst) + 1s * seg);
	return true;
}

std::pair<uint, uint> load_translation(uint seg, const fs::path& path, bool verify = false)
{
	std::ifstream cache(path);
	if (!cache.is_open()) {
		std::cerr << "Failed to open " << path << std::endl;
		return std::make_pair(0, 0);
	}

	// Extract cached translation
	line_id id{seg, 0};
	std::string temp, text;
	uint kept = 0;
	bool keep = true;
	bool is_broken = false;
	g_lines.segs[seg].prev_segs.clear();
	while (std::getline(cache, temp, '\n')) {
		// Skip prompt+example (isn't stored anymore, it's an artifact)
		if (temp.starts_with("SRC:")) {
			// TODO: properly verify file format and JP: line integrity
			continue;
		}
		if (temp.starts_with("PREV:")) {
			g_lines.segs[seg].prev_segs.emplace_back(std::move(temp)).erase(0, 5);
			continue;
		}
		if (temp.ends_with("\\")) {
			// TODO
			text.clear();
			continue;
		}
		text += temp;
		text += '\n';
		if (id != c_bad_id && temp.starts_with(iprefix)) {
			std::string expected = iprefix;
			expected += g_lines[id].name;
			expected += g_lines[id].text;
			if (temp != expected) {
				is_broken = true;
				if (g_mode == op_mode::print_info) {
					// Repair
					text.resize(text.size() - temp.size() - 1);
					text += expected;
					text += '\n';
				} else {
					// Consider broken, append everything to tail
					while (std::getline(cache, temp, '\n')) {
						text += temp;
						text += '\n';
					}
					break;
				}
			}
			while (std::getline(cache, temp, '\n')) {
				// Store both original and translated lines as a single string
				text += temp;
				text += '\n';
				if (!temp.starts_with(isuffix))
					continue;
				auto old = std::move(g_lines[id].tr_text);
				g_lines[id].tr_text = std::move(text);
				if (keep && g_lines[id].tr_text == old) {
					kept++;
				} else {
					g_lines[id].tr_tts.clear();
					keep = false;
				}
				g_lines.advance(id);
				break;
			}
		}
	}

	// Clear remaining lines
	for (auto id2 = id; id2 != c_bad_id; g_lines.advance(id2)) {
		if (!g_lines[id2].tr_text.empty()) {
			keep = false;
		}
		g_lines[id2].tr_text = {};
		g_lines[id2].tr_tts.clear();
	}

	// Check and load what remains in the file
	auto old = std::move(g_lines.segs[seg].tr_tail);
	g_lines.segs[seg].tr_tail = std::move(text);
	if (keep)
		kept = -1;

	return std::make_pair(id.second, verify ? !is_broken : kept);
}

void sigh(int) { g_stop = true; }

int main(int argc, char* argv[])
{
	if (std::getenv("VNSLEUTH_COREDUMP")) {
		rlimit lim{rlim_t(-1), rlim_t(-1)};
		setrlimit(RLIMIT_CORE, &lim);
	}

	g_mode = op_mode::rt_llama;
	if (argc == 2) {
		g_mode = op_mode::rt_cached;
	} else if (argc >= 3 && argv[2] == "--legacy"sv) {
		g_mode = op_mode::print_only;
	} else if (argc >= 3 && (argv[2] == "--check"sv || argv[2] == "--repair"sv)) {
		g_mode = op_mode::print_info;
	}

	static const auto argv_ = argv;
	static const auto print_usage = [](int, char**) {
		auto argv = argv_;
		std::cerr << "VNSleuth v0.5" << std::endl;
		std::cerr << "Usage 0: " << argv[0] << " <game_directory> --color" << std::endl;
		std::cerr << "\tUse existing translation cache only (only useful if it's complete)." << std::endl;
		std::cerr << "Usage 1: " << argv[0] << " <game_directory> --check" << std::endl;
		std::cerr << "\tPrint furigana, some information, initialize translation cache." << std::endl;
		std::cerr << "Usage 2: " << argv[0] << " <game_directory> --color -m <model> [<LLAMA args>...]" << std::endl;
		std::cerr << "\tStart translating in real time (recommended)." << std::endl;
		std::cerr << "Nothing to do." << std::endl;
	};

	if (argc < 2 || argv[1] == "--help"sv) {
		print_usage(argc, argv);
		return 1;
	}

	common_params params{};
	params.n_gpu_layers = 999;
	params.n_gpu_layers_draft = 999;
	params.flash_attn = 1; // FA is currently NEEDED for efficient GPU KV-cache saving/restoring
	//params.cache_type_k = "q8_0";
	//params.cache_type_v = "q8_0";
	params.n_ctx = 4096;
	params.n_batch = 2048;
	params.n_predict = 128;
	params.sparams.seed = 0;
	params.sparams.temp = 0.2;
	params.sparams.top_p = 0.3;
	params.sparams.penalty_last_n = 64;
	params.sparams.penalty_repeat = 1.1;
	params.warmup = false;
	params.model = ".";
	params.model_draft = "";
	params.n_keep = 6; // Used by background thread, number of lines to translate ahead of time

	// Basic param check
	std::string dir_name = argv[1];
	if (argc > 2) {
		argc--;
		argv++;
		if (g_mode == op_mode::print_only || g_mode == op_mode::print_info) {
			argc--;
			argv++;
		}
		if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE, print_usage)) {
			return 1;
		}

		// Disable colors
		if (!params.use_color)
			g_esc.disable();
		// Allow replacing translation prefix/suffix
		if (!params.input_prefix.empty())
			iprefix = std::move(params.input_prefix);
		if (!params.input_suffix.empty())
			isuffix = std::move(params.input_suffix);
		if (!params.prompt.empty()) {
			example = std::move(params.prompt);
			REPLACE(example, "\r", "");
			if (!example.ends_with("\n"))
				example += '\n';
		}
		params.n_draft = params.n_keep;
		if (g_mode == op_mode::rt_llama && params.model == ".")
			g_mode = op_mode::rt_cached;
	}

	// Parse scripts in a given directory
	std::string line, prompt_path;
	bool is_incomplete = false;
	bool is_broken = false;

	// Load file list recursively
	cache_prefix = fs::path(dir_name).filename();
	vnsleuth_path = fs::absolute(fs::path(dir_name)).lexically_normal() / "__vnsleuth";
	std::map<std::string, std::size_t, natural_order_less> file_list;
	if (fs::is_directory(dir_name)) {
		fs::create_directory(vnsleuth_path);
		for (const auto& entry : fs::recursive_directory_iterator(vnsleuth_path.parent_path(), fs::directory_options::follow_directory_symlink)) {
			if (entry.is_regular_file()) {
				// Skip __vnsleuth directory entirely
				const auto fname = entry.path().string();
				if (fname.starts_with(vnsleuth_path.c_str()))
					continue;
				// Check file size
				const auto size = entry.file_size();
				if (size > 10)
					file_list.emplace(entry.path(), size);
			}
		}
	} else if (fs::is_regular_file(dir_name) && dir_name.ends_with(".txt")) {
		// a.txt creates ./a/__vnsleuth/
		vnsleuth_path = fs::absolute(fs::path(dir_name - ".txt")).lexically_normal() / "__vnsleuth";
		fs::create_directories(vnsleuth_path);
	} else {
		std::cerr << "Not a directory: " << dir_name << std::endl;
		return 1;
	}

	replaces_path = vnsleuth_path / "__vnsleuth_replace.txt";
	prompt_path = vnsleuth_path / "__vnsleuth_prompt.txt";
	names_path = vnsleuth_path / "__vnsleuth_names.txt";

	// Open stat file for writing via memory map
	std::shared_ptr<void> stats_ptr;
	int stats_fd = open((vnsleuth_path / "__vnsleuth_stats").c_str(), O_CREAT | O_RDWR, 0666);
	if (stats_fd != -1 && ftruncate(stats_fd, alignof(vnsleuth_stats)) != -1) {
		stats_ptr = std::shared_ptr<void>(mmap(0, alignof(vnsleuth_stats), PROT_READ | PROT_WRITE, MAP_SHARED, stats_fd, 0), [=](void* ptr) {
			munmap(ptr, alignof(vnsleuth_stats));
			close(stats_fd);
		});
	} else {
		perror("Failed to create __vnsleuth_stats");
		stats_ptr = std::make_shared<vnsleuth_stats>();
	}
	g_stats = static_cast<vnsleuth_stats*>(stats_ptr.get());
	if (g_mode == op_mode::rt_llama) {
		ui64 z = 0;
		g_stats->start_time.compare_exchange_strong(z, std::time(nullptr));
	}

	if (prompt_path.empty() || !fs::is_regular_file(prompt_path)) {
		std::cerr << "Translation prompt not found: " << prompt_path << std::endl;
		if (!prompt_path.empty()) {
			// Create dummy prompt
			std::ofstream p(prompt_path, std::ios_base::trunc);
			if (p.is_open()) {
				p << "This is an accurate translation of a Japanese novel to English." << std::endl;
				p << "Line starting with En: only contains the translation of the preceding JP: line." << std::endl;
				p << "All honorifics are always preserved as -san, -chan, -sama, etc." << std::endl;
				p << "<START>" << std::endl;
				p.close();
				std::cerr << "Dummy translation prompt created: " << prompt_path << std::endl;
			}
		}
	}

	auto reload_names = [&]() -> bool {
		// Load (pre-translated) names
		std::lock_guard lock(g_mutex);

		std::ifstream names(names_path);
		if (names.is_open()) {
			auto old = std::move(g_dict);
			while (std::getline(names, line, '\n')) {
				REPLACE(line, "\r", "");
				if (const auto pos0 = line.find_first_of(":") + 1) {
					const auto pos1 = line.find_first_of(":", pos0);
					if (pos1 + 1 == 0) {
						// Untranslated name only
						if (pos0 != line.size()) {
							std::cerr << "Incorrect name formatting: " << line << std::endl;
							continue;
						}
						g_dict[std::move(line)];
						continue;
					}
					g_dict.emplace(line.substr(0, pos0), std::make_pair(line.substr(pos0, pos1 + 1 - pos0), line.substr(pos1 + 1)));
				} else {
					std::cerr << "Incorrect name formatting: " << line << std::endl;
				}
			}
			return g_dict != old;
		} else {
			std::cerr << "Translation names not found: " << names_path << std::endl;
			return false;
		}
	};

	if (!names_path.empty()) {
		reload_names();
	}

	auto reload_replaces = [&]() -> std::size_t {
		g_replaces = g_default_replaces;
		if (!replaces_path.empty()) {
			// Load replacement rules
			std::ifstream strs(replaces_path);
			if (strs.is_open()) {
				std::size_t count = 0;
				while (std::getline(strs, line, '\n')) {
					REPLACE(line, "\r", "");
					const bool a = std::count(line.cbegin(), line.cend(), ':') == 1;
					const bool b = std::count(line.cbegin(), line.cend(), '=') == 1;
					if (a ^ b) {
						const auto pos = line.find_first_of(a ? ":" : "=");
						std::string src(line.data(), pos);
						std::string dst(line.data() + pos + 1);
						// Override previous replace
						std::erase_if(g_replaces, [&](auto& pair) { return pair.first == src; });
						if (dst == src) {
							// No-op replace
						} else {
							g_replaces.emplace_back(std::move(src), std::move(dst));
							count++;
						}
					} else {
						std::cerr << "Failed to parse replacement string: " << line << std::endl;
						continue;
					}
				}

				return count;
			}
		}

		return 0;
	};

	// Analyze ExHIBIT script files to obtain "encryption" key
	{
		std::vector<std::string> to_analyze;
		for (auto& [name, size] : file_list) {
			if (name.ends_with(".rld")) {
				std::string data;
				data.resize(size > 0xffd0 ? 0xffd0 : size);
				std::ifstream f(name, std::ios::binary);
				if (f.is_open() && f.read(data.data(), data.size())) {
					if (data.starts_with("\0DLR"sv)) {
						to_analyze.emplace_back(std::move(data));
					}
				} else {
					std::cerr << "Failed to read file: " << name << std::endl;
					return 1;
				}
			}
		}

		std::array<uint, 1024 / 4> _key;
		std::bitset<1024 / 4> conflict{};
		for (uint i = 0; i < _key.size(); i++) {
			uint bytes = 0;
			// Raw bytes -> counter
			std::unordered_map<uint, uint> stats;
			for (auto& s : to_analyze) {
				for (uint j = i * 4 + 16; j + 3 < s.size(); j += sizeof(_key)) {
					std::memcpy(&bytes, s.data() + j, 4);
					stats[bytes]++;
				}
			}
			uint _max = 0;
			for (auto& [seq, count] : stats) {
				if (count > _max) {
					bytes = seq;
					_max = count;
				}
			}
			uint sure = -1;
			for (auto& [seq, count] : stats) {
				if (count < _max)
					sure = std::min(sure, _max - count);
				if (count == _max && seq != bytes) {
					conflict.set(i);
					sure = 0;
				}
			}
			if (sure < 10) {
				std::cerr << "ExHIBIT: Key is ambiguous at position " << i * 4 << "+16" << std::endl;
				conflict.set(i);
			} else {
				_key[i] = bytes;
			}
		}
		if (conflict.none() && _key.size() - std::count(_key.begin(), _key.end(), 0)) {
			std::fprintf(stderr, "ExHIBIT: key found: ");
			for (auto& key : _key)
				std::fprintf(stderr, "%08x", __builtin_bswap32(key));
			std::fprintf(stderr, "\n");
			extern char g_exhibit_key[1024];
			std::memcpy(g_exhibit_key, _key.data(), sizeof(g_exhibit_key));
		}
	}

	if (dir_name.ends_with(".txt") && file_list.empty()) {
		// Load a single raw text file
		std::ifstream txt(dir_name, std::ios::binary);
		std::string data(fs::file_size(dir_name) + 1, '\n');
		txt.read(data.data(), data.size() - 1);
		auto& seg = g_lines.segs.emplace_back();
		g_lines.segs_by_name[".txt"] = 0;
		seg.src_name = ".txt";
		seg.cache_name = "__vnsleuth_text.txt";
		std::vector<std::u32string> lines{1};
		std::u32string all_lines = to_utf32(data);
		const bool no_squash = std::getenv("VNSLEUTH_NO_SQUASH");
		for (std::size_t p = 0; p < all_lines.size();) {
			const std::size_t end = all_lines.find_first_of(U"\n✏", p);
			std::u32string line = all_lines.substr(p, end - p);
			p = end + 1;
			REPLACE(line, U"・・・", U"・・"); // Preprocess dot sequence (accent)
			REPLACE(line, U"（・・）", U"");   // Remove accent furigana
			REPLACE(line, U"《・・》", U"");
			REPLACE(line, U"(・・)", U"");
			REPLACE(line, U"（）", U""); // Remove empty furigana from certain formats
			REPLACE(line, U"《》", U"");
			REPLACE(line, U"()", U"");
			// Trim spaces
			line.resize(line.find_last_not_of(U" 　\t\r") + 1);
			line.erase(0, line.find_first_not_of(U" 　\t"));
			if (line.empty()) {
				continue;
			}
			REPLACE(line, U"  ", U" ");
			REPLACE(line, U"　　", U"　");
			if (line.starts_with(U"<") && line.ends_with(U">")) {
				// TODO: skip html
				continue;
			}
			static auto is_jp = [](char32_t c) -> bool {
				// TODO: revise
				if ((c >= 0x3040 && c <= 0x30ff) || // Hiragana, katakana
					(c >= 0x4e00 && c <= 0x9fff) || // CJK ideograms
					(c >= 0xf900 && c <= 0xfaff) || //
					(c >= 0xff65 && c <= 0xff9f))	// Halfwidth kana
					return true;
				return false;
			};
			// Remove padding ASCII spaces within Japanese text
			for (std::size_t i = 0; i < line.size();) {
				if (line[i] == ' ' && i) {
					if (is_jp(line[i - 1]) || is_jp(line[i + 1])) {
						line.erase(i, 1);
						continue;
					}
				}
				// Transform certain full-width character to ASCII (due to anecdotal improvement observation)
				if (line[i] == U'　' && i && !is_jp(line[i - 1]) /*&& !is_jp(line[i + 1])*/)
					line[i] = ' ';
				if (line[i] == U'、' && i && !is_jp(line[i - 1]))
					line[i] = ',';
				if (line[i] == U'．' && i && !is_jp(line[i - 1]))
					line[i] = '.';
				if (line[i] == U'＇' && i && !is_jp(line[i - 1]))
					line[i] = '\'';
				if (line[i] == U'？' && i && !is_jp(line[i - 1]) && line[i - 1] != U'？' && line[i - 1] != U'！')
					line[i] = '?';
				if (line[i] == U'！' && i && !is_jp(line[i - 1]) && line[i - 1] != U'？' && line[i - 1] != U'！')
					line[i] = '!';
				if (line[i] == U'：' && i && !is_jp(line[i - 1]))
					line[i] = ':';
				if (line[i] >= U'０' && line[i] <= U'９')
					line[i] -= U'０' - '0';
				if (line[i] >= U'Ａ' && line[i] <= U'Ｚ')
					line[i] -= U'Ａ' - 'A';
				if (line[i] >= U'ａ' && line[i] <= U'ｚ')
					line[i] -= U'ａ' - 'a';
				i++;
			}
			// Function to split too long strings
			auto new_line = [&](std::u32string str) {
				// When includes split indicator, real max_size is (max_size + 1)
				static constexpr std::size_t max_size = 1023;
				while (lines.back().size() > max_size) {
					std::size_t new_size = max_size;
					// Find convenient split position (dot or comma)
					for (auto delim : {U'。', U'、', U'.', U';', U','}) {
						if (auto pos = lines.back().rfind(delim, max_size - 1) + 1) {
							new_size = pos;
							break;
						}
					}
					// Trim starting spaces after split position
					auto next = lines.back().substr(lines.back().find_first_not_of(U"　 \t", new_size));
					lines.back().resize(new_size);
					lines.back() += U'↴';
					lines.emplace_back(std::move(next));
				}
				lines.emplace_back(std::move(str));
			};
			if (!line.empty() && !no_squash) {
				// Check line squashing conditions
				static constexpr std::array<char32_t, 3> braces[]{
					U"「」", //
					U"『』", //
					U"【】", //
					U"《》", //
					U"〝〟", //
				};
				static auto is_open = [](std::u32string_view str) -> int {
					std::u32string stack;
					for (auto& c : str) {
						for (auto& [open, close, _] : braces) {
							if (c == open) {
								stack.push_back(close);
							} else if (c == close) {
								if (!stack.ends_with(close))
									return -1;
								stack.pop_back();
							}
						}
					}
					return !stack.empty();
				};
				static auto has_speech = [](std::u32string_view str) -> bool {
					for (auto& arr : braces) {
						if (str.find_first_of(arr.data()) + 1)
							return true;
					}
					return false;
				};
				static auto start_speech = [](std::u32string_view str) -> bool {
					for (auto& [open, close, _] : braces) {
						if (str.starts_with(open))
							return true;
					}
					return false;
				};
				static auto end_speech = [](std::u32string_view str) -> bool {
					for (auto& [open, close, _] : braces) {
						if (str.ends_with(close))
							return true;
					}
					return false;
				};
				const int prev_op = is_open(lines.back());
				if (has_speech(lines.back()) && prev_op == 0) {
					// Don't squash two speeches
					// Also don't squash with lines not starting with text (TODO: fix jp-specific check)
					if (start_speech(lines.back()) && end_speech(lines.back()) && is_jp(line[0])) {
						if (is_open(line) == 0 && !start_speech(line)) {
							lines.back() += line;
							// Add sentinel to prevent further squashing
							new_line(U"");
							continue;
						}
					}
				} else if (prev_op > 0) {
					// Squash split speech
					lines.back() += U"　";
					lines.back() += line;
					continue;
				} else if (lines.back().ends_with(U'、') || lines.back().ends_with(',')) {
					// Append speech after incomplete sentence
					if (prev_op == 0 && start_speech(line)) {
						lines.back() += line;
						continue;
					}
				}
			}
			new_line(std::move(line));
			if (p >= all_lines.size())
				new_line(U"");
		}
		for (auto& line : lines) {
			if (!line.empty()) {
				extern void add_line(int choice, std::string name, std::string text);
				add_line(0, "", to_utf8(line));
			}
		}
		const auto cache_path = vnsleuth_path / seg.cache_name;
		if (fs::is_regular_file(cache_path)) {
			const auto [tr_lines, verified] = load_translation(0, cache_path, true);
			if (!verified) {
				is_incomplete = true;
				is_broken = true;
			} else if (seg.tr_tail.empty() || tr_lines < seg.lines.size()) {
				is_incomplete = true;
			}
		} else {
			is_incomplete = true;
		}
	}

	for (auto& [fname, fsize] : file_list) {
		const int fd = open(fname.c_str(), O_RDONLY);
		if (fd >= 0) {
			// Map file in memory (alloc 1 more zero byte, like std::string)
			const auto map_sz = fsize + 1;
			const std::shared_ptr<void> file(mmap(0, map_sz, PROT_READ, MAP_SHARED, fd, 0), [=](void* ptr) {
				munmap(ptr, map_sz);
				close(fd);
			});

			// View file as chars
			script_parser data{{static_cast<char*>(file.get()), fsize}};
			std::string name = fs::path(fname).filename();
			for (std::size_t x = g_lines.segs.size(), j = (data.read_segments(name), x); j < g_lines.segs.size();) {
				auto& new_seg = g_lines.segs[j];
				if (new_seg.lines.empty()) {
					// Remove empty segments (should really only be last in `segs`)
					g_lines.segs_by_name.erase(new_seg.src_name);
					g_lines.segs.erase(g_lines.segs.begin() + j);
					break;
				}
				if (g_mode == op_mode::print_info || g_mode == op_mode::rt_cached || g_mode == op_mode::rt_llama) {
					if (g_mode == op_mode::print_info) {
						std::cerr << "Found data: " << new_seg.src_name << std::endl;
						std::cerr << "Cache file: " << new_seg.cache_name;
					}
					auto cache_path = vnsleuth_path / new_seg.cache_name;
					if (fs::is_regular_file(cache_path)) {
						const auto [tr_lines, verified] = load_translation(j, cache_path, true);
						if (!verified) {
							is_incomplete = true;
							is_broken = true;
							if (g_mode == op_mode::print_info)
								std::cerr << " (broken)" << std::endl;
						} else if (new_seg.tr_tail.empty() || tr_lines < new_seg.lines.size()) {
							is_incomplete = true;
							if (g_mode == op_mode::print_info)
								std::cerr << " (partial)" << std::endl;
						} else {
							if (g_mode == op_mode::print_info)
								std::cerr << " (full)" << std::endl;
						}
					} else {
						if (g_mode == op_mode::print_info)
							std::cerr << " (not found)" << std::endl;
						is_incomplete = true;
					}
				}
				j++;
			}
		} else {
			std::cerr << "Error: Could not open file: " << fname << std::endl;
		}
	}

	std::cerr << "Loaded files: " << g_lines.segs.size() << std::endl;
	std::cerr << "Loaded lines: " << g_lines.count_lines() << std::endl;
	std::cerr << "Loaded names: " << g_dict.size() - 1 << std::endl;
	std::cerr << "Loaded replaces: " << reload_replaces() << std::endl;
	if (is_incomplete) {
		if (g_mode == op_mode::print_info || g_mode == op_mode::rt_cached)
			std::cerr << "Translation is incomplete." << std::endl;
	}
	if (is_broken) {
		std::cerr << "Some cache files are damaged." << std::endl;
	}

	if (g_mode == op_mode::print_info) {
		// Print known furigana
		for (auto&& [type_as, read_as] : g_furigana) {
			std::cerr << type_as << "=" << read_as << std::endl;
		}

		// Update names
		if (!g_lines.segs.empty() && g_dict.size() > 1) {
			update_names(names_path);
		}

		if (argv[0] == "--repair"sv) {
			// Update all segments
			for (auto& seg : g_lines.segs) {
				if (!seg.lines.at(0).tr_text.empty() || !seg.tr_tail.empty()) {
					update_segment(&seg - g_lines.segs.data(), false);
				}
			}
		}

		if (!g_lines.segs.empty() && argv[0] == "--check"sv) {
			// Dump text
			std::ofstream dump(vnsleuth_path / "__vnsleuth_dump.txt", std::ios::trunc);
			for (auto& line : g_lines) {
				dump << line.name << line.text << std::endl;
			}
		}

		return 0;
	}

	if (is_broken) {
		return 1;
	}

	auto reload_prompt = [&]() -> bool {
		// Load prompt
		const auto old_prompt = std::move(params.prompt);

		std::ifstream p(prompt_path);
		if (p.is_open()) {
			params.prompt = {};
			char c{};
			while (p.get(c))
				params.prompt += c;

			// Preprocess
			REPLACE(params.prompt, "\r", "");
			if (!params.prompt.ends_with("\n"))
				params.prompt += '\n';

			// Add optional dictionary from name annotations
			params.prompt -= "<START>\n";
			for (auto& [_, pair] : g_dict) {
				if (!pair.second.empty()) {
					params.prompt += "Dictionary:\n";
					break;
				}
			}
			for (auto& [orig, pair] : g_dict) {
				auto& [tran, ann] = pair;
				if (!ann.empty()) {
					params.prompt += orig - ":";
					params.prompt += '\t';
					params.prompt += tran - ":";
					params.prompt += '\t';
					params.prompt += ann;
					params.prompt += '\n';
				}
			}
			params.prompt += "<START>\n";
		} else {
			std::cerr << "Failed to load prompt: " << prompt_path << std::endl;
			return true;
		}

		// Check if the prompt has been modified
		return params.prompt != old_prompt;
	};
	reload_prompt();
	if (argc > 2) {
		if (g_mode == op_mode::rt_llama) {
			// Initialize llama.cpp
			std::cerr << "Preparing translator..." << std::endl;
			if (!translate(params, c_bad_id))
				return 1;
		}
	}

	if (true) {
		std::cerr << "Waiting for input..." << std::endl;
		// Hack to prevent interleaving stderr/stdout
		usleep(300'000);
	}

	line_id next_id = c_bad_id;
	line_id prev_id = c_bad_id;

	// Load history given specific id
	auto load_history = [&](line_id last) {
		g_history.clear();
		if (last == c_bad_id)
			return;

		std::deque<uint> seg_list;
		std::unordered_map<segment_info*, std::size_t> seg_rpos;

		seg_list.push_front(last.first);
		while (true) {
			auto& seg = g_lines.segs.at(seg_list.front());
			if (auto max = seg.prev_segs.size()) {
				auto rpos = seg_rpos[&seg]++;
				if (rpos < max) {
					auto& name = seg.prev_segs[max - 1 - rpos];
					auto found = g_lines.segs_by_name.find(name);
					if (found == g_lines.segs_by_name.end()) {
						throw std::runtime_error("Invalid history segment name: " + name);
					} else {
						seg_list.push_front(found->second);
						continue;
					}
				}
			}
			break;
		}

		for (uint& s : seg_list) {
			auto& seg = g_lines.segs.at(s);
			for (uint i = 0; i < seg.lines.size(); i++) {
				if (g_history.emplace_back(s, i) == last && &s == &seg_list.back())
					break;
			}
		}

		std::cerr << "Loaded history: " << g_history.size() << std::endl;
	};

	std::unordered_map<uint, std::pair<line_id, line_id>> incompletes;
	auto get_incompletes = [&](bool use_res = true) -> std::pair<line_id, line_id> {
		auto result1 = c_bad_id;
		auto result2 = c_bad_id;
		incompletes.clear();
		fs::file_time_type last_time{};
		last_time = last_time.min();
		for (auto& s : g_lines.segs) {
			// Skip untranslated segments
			if (s.lines.at(0).tr_text.empty() && s.prev_segs.empty())
				continue;
			// Use lack of finalization to determine candidate
			if (s.tr_tail.empty() || s.lines.back().tr_text.empty()) {
				auto& [last, next] = incompletes[&s - g_lines.segs.data()];
				last.first = &s - g_lines.segs.data();
				last.second = -1 + std::count_if(s.lines.begin(), s.lines.end(), [](line_info& line) {
					// Count translated lines
					return !line.tr_text.empty();
				});
				next = g_lines.next(last);
				if (last.second + 1 == 0) {
					// Inbetween segments (TODO: seems broken)
					next.first = last.first;
					next.second = 0;
					last.first = g_lines.segs_by_name.at(s.prev_segs.back());
					last.second = g_lines.segs[last.first].lines.size() - 1;
				}

				// Check mtime to detect latest
				if (!use_res)
					continue;
				std::error_code ec{};
				auto time = fs::last_write_time(vnsleuth_path / s.cache_name, ec);
				if (!ec && time >= last_time) {
					result1 = last;
					result2 = next;
					last_time = time;
				}
			}
		}

		// Return latest translated line and next id
		return std::make_pair(result1, result2);
	};
	std::tie(prev_id, next_id) = get_incompletes();

	// List of IDs that must be processed (possibly more than once)
	std::vector<line_id> id_queue;
	auto flush_id_queue = [&]() -> bool {
		if (g_mode == op_mode::print_only) {
			for (auto& id : id_queue) {
				if (static bool prompt_sent = false; !std::exchange(prompt_sent, true)) {
					// Send prompt and example before first line
					std::string out = params.prompt + example;
					for (uint i = 0; i < out.size(); i++) {
						if (out[i] == '\n') {
							out.insert(i, "\\");
							i++;
						}
					}
					std::cout << out << std::flush;
				}
				print_line(id);
			}
			id_queue.clear();
			return true;
		}

		bool need_kick = false;
		for (auto& id : id_queue) {
			if (g_stop) {
				need_kick = false;
				break;
			}
			std::unique_lock lock(g_mutex);
			auto& line = g_lines[id];
			if (line.tr_text.empty()) {
				if (g_mode != op_mode::rt_llama) {
					std::cerr << "Error: Translation cache is incomplete" << std::endl;
					break;
				}
				lock.unlock();
				// Wait for current id instead of discarding it
				if (!translate(params, g_lines.next(id), tr_cmd::sync))
					return false;
			}
			if (line.tr_text.empty()) {
				need_kick = true;
				if (!translate(params, id))
					return false;
			} else {
				auto out = line.tr_text; // copy

				static const auto real_iprefix = iprefix + (iprefix.ends_with(" ") ? "" : " ");
				static const auto alt_iprefix = "\n" + real_iprefix;
				if (!out.starts_with(real_iprefix)) {
					if (auto pref_pos = out.find(alt_iprefix) + 1) {
						out.erase(pref_pos, alt_iprefix.size() - 1);
						out.replace(out.find(line.text, pref_pos), line.text.size(), apply_replaces(line.text, false, 0));
						out.insert(0, g_esc.orig);
					} else {
						std::cerr << iprefix << " not found in translation cache." << std::endl;
						return false;
					}
				} else {
					out.replace(0, iprefix.size(), g_esc.orig);
					out.replace(out.find(line.text), line.text.size(), apply_replaces(line.text, false, 0));
				}

				static const auto real_isuffix = "\n" + isuffix + (isuffix.ends_with(" ") ? "" : " ");
				const auto suff_pos = out.find(real_isuffix);
				if (suff_pos + 1 == 0) {
					std::cerr << isuffix << " not found in translation cache." << std::endl;
					return false;
				}
				out.replace(suff_pos + 1, real_isuffix.size() - 1, g_esc.tran);
				std::cout << out << g_esc.reset << std::flush;
				if (lock)
					lock.unlock();

				need_kick = true;
				if (g_mode == op_mode::rt_llama && g_lines.is_last(id)) {
					need_kick = false;
					// Prepare to autosave segment on last id
					if (!translate(params, line_id{id.first, 0u - 1}, tr_cmd::sync))
						return false;
				}
				if (g_mode == op_mode::rt_llama && g_history.back() == id) {
					g_stats->raw_accepts++;
				}
			}
			if (g_mode == op_mode::rt_llama && id == g_history.back() && id.second % 30 == 29) {
				// Autosave lines at each 30th line
				update_segment(id.first, true, id.second + 1);
			}
		}

		if (g_mode != op_mode::rt_llama) {
			id_queue.clear();
		} else if (need_kick) {
			if (!translate(params, id_queue.back(), tr_cmd::kick))
				return false;
		}
		return true;
	};

	std::u16string last_line;
	if (g_mode == op_mode::rt_llama) {
		// Show previous message(s) in a limited fashion (one message + selections)
		load_history(prev_id);
		auto it = g_history.rbegin();
		if (it != g_history.rend()) {
			last_line = g_lines[*it].sq_text;
			id_queue = {*it};
			while (g_lines[id_queue[0]].name.starts_with("選択肢#")) {
				it++;
				if (it != g_history.rend()) {
					id_queue.insert(id_queue.begin(), *it);
				} else {
					break;
				}
			}
			if (!flush_id_queue())
				return 1;
		}
	}

	auto last_input_time = std::chrono::steady_clock::now();
	auto update_input_time = [&]() {
		if (g_mode != op_mode::rt_llama) {
			return;
		}
		g_stats->last_time = std::time(nullptr);
		auto _now = std::chrono::steady_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(_now - last_input_time);
		if (diff < 1min) {
			// Approximately actual reading time
			g_stats->rt_reading_ms += diff.count();
		} else {
			// Otherwise, probably AFK time
			auto mins = std::chrono::duration_cast<std::chrono::minutes>(_now - last_input_time);
			uint pos = 31 - std::countl_zero<uint>(mins.count());
			g_stats->rt_afk_ms[pos] += diff.count();
		}
		last_input_time = _now;
	};

	// Setup Ctrl+C handler
	signal(SIGINT, sigh);

	// When line is found and is expected, it's printed out (back_buf remains empty).
	// When line is found and is unique, the back buffer is flushed as well.
	// When line is found but not unique, it's added to the back buffer.
	// Previous non-unique lines are printed if they exactly precede unique line, otherwise wiped.
	std::vector<std::u16string> back_buf;
	while (!g_stop && std::getline(std::cin, line)) {
		update_input_time();
		if (g_stop)
			break;
		if (g_mode == op_mode::rt_llama) {
			if (line.empty() || line.starts_with("\01")) {
				// Process rewrite request: only last line is supported currently (TODO)
				while (line.starts_with("\01")) {
					line.erase(0, 1);
				}
				if (id_queue.empty() || id_queue.back() != g_history.back()) {
					std::cerr << "Cannot process rewrite request here. Try rewinding (^B)." << std::endl;
					continue;
				}
				if (!translate(params, g_lines.next(id_queue.back()), tr_cmd::sync))
					return 1;
				g_neg.clear();
				auto& tr_text = g_lines[id_queue.back()].tr_text;
				if (tr_text.rfind(line) + 1 > tr_text.rfind("\n" + isuffix) + 1 + isuffix.size() + !isuffix.ends_with(" "))
					g_neg = std::move(line);
				if (!translate(params, id_queue.back()))
					return 1;
				g_neg.clear();
				g_stats->rt_rewrites++;
				if (!translate(params, id_queue.back(), tr_cmd::kick))
					return 1;
				continue;
			} else if (line == "\02" || line == "\b") {
				// Process rewind request (^B)
				// Remove last history entry, show previous one
				if (g_history.size() < 2) {
					continue;
				}

				std::cerr << "Rewinding... " << std::endl;
				next_id = g_history.back();
				g_history.pop_back();
				prev_id = g_history.back();
				id_queue = {prev_id};
				if (g_history.back().second == 0) {
					// TODO: this is unsafe because it can break history chain, this assumes forward request will follow
					std::cerr << "*** Going back to segment " << g_lines.segs[prev_id.first].src_name << std::endl;
				}
				last_line.clear();
				if (!translate(params, g_lines.next(next_id), tr_cmd::sync))
					return 1;
				if (!translate(params, {0u, 1u}, tr_cmd::eject))
					return 1;
				g_lines[next_id].tr_text.clear();
				g_lines[next_id].tr_tts.clear();
				if (!flush_id_queue())
					return 1;
				continue;
			} else if (line == "\x0e") {
				// Process forward ("next") request (^N)
				if (g_history.empty()) {
					prev_id = c_bad_id;
					next_id = {0, 0};
				} else if (g_lines.is_last(g_history.back())) {
					prev_id = g_history.back();
					if (prev_id.first == g_lines.segs.size() - 1) {
						next_id = c_bad_id;
						continue;
					}
					next_id.first = prev_id.first + 1;
					next_id.second = 0;
				} else {
					prev_id = g_history.back();
					next_id = g_lines.next(prev_id);
				}

				if (next_id.second == 0)
					std::cerr << g_esc.reset << "*** Forced segment: " << g_lines.segs[next_id.first].src_name << std::endl;
				last_line.clear();
				line = g_lines[next_id].text;
				// Passthrough
			} else if (line == "\06") {
				// Process finalize request (^F)
				if (g_history.empty() || prev_id != g_history.back() || !g_lines.is_last(prev_id)) {
					std::cerr << "Must be at the end of segment, try ^N." << std::endl;
					continue;
				}
				if (file_list.empty()) {
					std::cerr << "Finalization disabled for single text file." << std::endl;
					continue;
				}

				if (!translate(params, c_bad_id, tr_cmd::sync))
					return 1;
				if (!g_lines.segs[prev_id.first].tr_tail.empty()) {
					std::cerr << "Segment already finalized, try manual fix." << std::endl;
					continue;
				}

				g_lines.segs[prev_id.first].tr_tail = "<END>\n";
				update_segment(prev_id.first);
				prev_id = c_bad_id;
				next_id = g_history.front();
				id_queue.clear();
				g_history.clear();
				if (!translate(params, c_bad_id, tr_cmd::reload))
					return 1;
				continue;
			} else if (line == "\03") {
				// Process terminate request (^C)
				g_stop = true;
				break;
			} else if (line == "\x12") {
				// Process reload request (^R)
				if (!translate(params, {0, 0}, tr_cmd::sync))
					return 1;
				reload_names();
				reload_replaces();
				bool full_reload = reload_prompt();
				uint kept_lines = -2;
				for (auto& seg : g_lines.segs) {
					std::lock_guard lock{g_mutex};
					const uint segment = &seg - g_lines.segs.data();
					if (!seg.lines[0].tr_text.empty()) {
						auto [loaded, kept] = load_translation(segment, vnsleuth_path / seg.cache_name);
						if (segment == next_id.first) {
							kept_lines = kept;
						} else if (kept + 1) {
							full_reload = true;
						}
					}
				}

				if (kept_lines == 0u - 2)
					full_reload = true;

				std::tie(prev_id, next_id) = get_incompletes();
				if (prev_id != c_bad_id && !full_reload) {
					if (id_queue.empty() || id_queue.back().first != prev_id.first || id_queue.back() < prev_id)
						full_reload = true;
				}
				if (prev_id == c_bad_id && !id_queue.empty()) {
					// Restore prev_id
					prev_id = id_queue.back();
					next_id = g_lines.next(prev_id);
				}

				if (full_reload) {
					std::cerr << "Reloading cache..." << std::endl;
					load_history(prev_id);
					if (!translate(params, c_bad_id, tr_cmd::reload))
						return 1;
					if (!id_queue.empty() && prev_id != c_bad_id && prev_id.first != id_queue.back().first) {
						id_queue.clear();
					}
					std::cerr << "Cache reloaded." << std::endl << std::flush;
				} else if (kept_lines + 1) {
					// Optimized reload
					prev_id = g_history.back();
					const uint to_eject = prev_id.second + 1 - kept_lines;
					if (params.verbosity)
						std::fprintf(stderr, "%s[] Reload: eject %u, keep %u\n", g_esc.reset, to_eject, kept_lines);
					if (!translate(params, {0u, to_eject}, tr_cmd::eject))
						return 1;
					if (!translate(params, {next_id.first, kept_lines}, tr_cmd::reload))
						return 1;
				}

				if (!id_queue.empty()) {
					// Restore prev_id
					prev_id = id_queue.back();
					next_id = g_lines.next(prev_id);
				}
				if (prev_id != c_bad_id) {
					// Edge case of reloading
					if (!translate(params, prev_id, tr_cmd::kick))
						return 1;
				}
				if (!flush_id_queue())
					return 1;
				g_stats->rt_reloads++;
				continue;
			} else if (line == "\x13") {
				// Process stop-and-save request (^S)
				if (!translate(params, next_id, tr_cmd::sync))
					return 1;
				if (prev_id != c_bad_id && update_segment(prev_id.first)) {
					std::cerr << "Wrote file: " << g_lines.segs[prev_id.first].cache_name << std::endl;
				}
				continue;
			} else if (file_list.empty()) {
				// Disable clipboard input
				continue;
			}
		}

		std::u16string sq_line = squeeze_line(line);
		// Erase all characters that aren't encountered in scripts
		std::erase_if(sq_line, [](char16_t c) { return !g_chars[c]; });
		// Squeeze again
		sq_line.erase(std::unique(sq_line.begin(), sq_line.end()), sq_line.end());
		// Levenshtein distance must fit in a single byte in this implementation
		if (sq_line.empty() || sq_line == last_line || (!file_list.empty() && sq_line.size() > 255)) {
			continue;
		}
		auto it = g_strings.find(sq_line);
		bool force_ambi = false;
		if (next_id != c_bad_id && (it != g_strings.end() && it->second != next_id)) {
			// Exact mismatch has no definitive solution because sq_line can be still close to next_id
			const auto& sq = g_lines[next_id].sq_text;
			if (!file_list.empty() && levenshtein_distance(sq_line, sq) <= std::max<uint>(3, sq.size() * 1 / 3)) {
				// TODO: back_buf implementation is outdated, prefer exact match for now
				//force_ambi = true;
				if (it->second != c_bad_id) {
					std::cerr << "Warning: ambiguous line: " << line << std::endl;
				}
			}
		}
		if (next_id != c_bad_id && (it == g_strings.end() || it->second == c_bad_id)) {
			// Fuzzy match of predicted next line
			const auto& sq = g_lines[next_id].sq_text;
			if (!file_list.empty() && levenshtein_distance(sq_line, sq) <= sq.size() * 2 / 3) {
				sq_line = sq;
				it = g_strings.find(sq_line);
			}
		}
		if (next_id == c_bad_id && (it == g_strings.end() || it->second == c_bad_id) && back_buf.empty()) {
			// For smoother startup from an ambiguous line, attempt to show first line in a segment
			const auto it2 = g_start_strings.find(sq_line);
			if (it2 != g_start_strings.end()) {
				next_id = it2->second;
			} else {
				for (const auto& [sq, id] : g_start_strings) {
					// Fuzzy match fallback
					if (levenshtein_distance(sq_line, sq) <= sq.size() * 2 / 3) {
						if (next_id != c_bad_id) {
							// Multiple matches, abort search
							next_id = c_bad_id;
							break;
						}
						next_id = id;
					}
				}
				if (next_id != c_bad_id) {
					sq_line = g_lines[next_id].sq_text;
					it = g_strings.find(sq_line);
				}
			}
		}
		if (!force_ambi && it == g_strings.end()) {
			// Fallback full fuzzy search (naïve, bruteforce)
			static const auto sorted_sqstrings = []() -> std::vector<std::u16string_view> {
				// List squeezed strings
				std::vector<std::u16string_view> result;
				result.reserve(g_strings.size());
				for (auto& [sq, _] : g_strings) {
					result.push_back(sq);
				}
				// Sort so bigger ones go first
				std::sort(result.begin(), result.end(), [](auto a, auto b) {
					if (a.size() > b.size())
						return true;
					if (a.size() < b.size())
						return false;
					return a < b;
				});
				for (uint j = 0; j < result.size(); j++) {
					for (uint i = 0; i < std::min<std::size_t>(255, result[j].size()); i++) {
						s_char_columns[i].resize(result.size());
						s_char_columns[i][j] = result[j][i];
					}
				}
				return result;
			}();
			std::u16string_view found{};
			// uint last = -1;
			// uint runs = 0;
			// auto stamp0 = std::chrono::steady_clock::now();
			// for (const auto& sq : sorted_sqstrings) {
			// 	// Optimizations: skip strings if size difference exceeds limit
			// 	if (sq.size() > sq_line.size() * 3 / 2 || sq.size() < sq_line.size() * 2 / 3)
			// 		continue;
			// 	if (sq.size() > sq_line.size() && sq.size() - sq_line.size() > last)
			// 		continue;
			// 	if (sq.size() < sq_line.size() && sq_line.size() - sq.size() > last)
			// 		continue;
			// 	runs++;
			// 	uint next = levenshtein_distance(sq_line, sq);
			// 	if (next > sq.size() * 2 / 3)
			// 		continue;
			// 	if (next == last) {
			// 		// Ambiguous (TODO?)
			// 		found = {};
			// 	}
			// 	if (next < last) {
			// 		last = next;
			// 		found = sq;
			// 	}
			// }
			auto stamp1 = std::chrono::steady_clock::now();
			std::span<const std::u16string_view> span = sorted_sqstrings;
			while (!span.empty() && span.back().size() < sq_line.size() * 2 / 3)
				span = span.subspan(0, span.size() - 1);
			std::size_t skip = 0;
			while (skip < span.size() && span[skip].size() > sq_line.size() * 3 / 2)
				skip++;
			auto result = levenshtein_distance(skip, span.subspan(skip), sq_line, sq_line.size() / 2 + 1);
			if (result.size() == 1) {
				// TODO: handle multiple results
				found = result[0];
			}
			auto stamp2 = std::chrono::steady_clock::now();
			// auto time_ns = 0ns + stamp1 - stamp0;
			// std::cerr << "Full search took " << time_ns.count() / 1e6 << "ms, Len=" << sq_line.size() << ", runs=" << runs << std::endl;
			auto time_ns2 = 0ns + stamp2 - stamp1;
			std::cerr << "Batch search took " << time_ns2.count() / 1e6 << "ms, Span=" << span.size() - skip << " R=" << result.size() << std::endl;
			if (!found.empty()) {
				sq_line = found;
				it = g_strings.find(sq_line);
			}
		}
		if (!force_ambi && it == g_strings.end()) {
			if (params.verbosity)
				std::fprintf(stderr, "String rejected: %s\n", line.c_str());
			continue;
		}
		last_line = sq_line;
		if (g_mode == op_mode::rt_llama) {
			id_queue.clear();
		}
		line_id last_id = next_id;
		if (last_id == c_bad_id || g_lines[last_id].sq_text != sq_line) {
			// Handle unexpected line:
			if (force_ambi || it->second == c_bad_id) {
				// Remember ambiguous line
				std::cerr << g_esc.buf << "Buffered: " << line << g_esc.reset << std::endl;
				back_buf.emplace_back(std::move(sq_line));
				next_id = c_bad_id;
				continue;
			}

			if (it->second == prev_id) {
				// Avoid repeating previous line
				back_buf.clear();
				next_id = g_lines.next(prev_id);
				continue;
			}

			next_id = it->second;
			last_id = it->second;

			// Restore back log:
			for (auto itr = back_buf.rbegin(); itr != back_buf.rend(); itr++) {
				bool found = false;
				// Compensate for repetition suppression
				while (next_id.second) {
					next_id.second--;
					if (g_lines[next_id].sq_text == *itr) {
						found = true;
					} else {
						next_id.second++;
						break;
					}
				}
				if (!found)
					break;
			}

			back_buf.clear();
		}

		// Handle forward jump within segment (enqueue all preceding lines)
		if (g_history.empty()) {
			if (next_id != c_bad_id && g_mode == op_mode::rt_llama) {
				// Try to load history from arbitrary location
				if (next_id.second) {
					prev_id = next_id;
					prev_id.second--;
					if (!g_lines[prev_id].tr_text.empty())
						load_history(prev_id);
					else {
						// Fallback
						prev_id = c_bad_id;
						next_id.second = 0;
					}
				} else if (auto& seg = g_lines.segs[next_id.first]; !seg.prev_segs.empty()) {
					auto seg_id = g_lines.segs_by_name[seg.prev_segs.back()];
					prev_id.first = seg_id;
					prev_id.second = g_lines.segs[seg_id].lines.size() - 1;
					load_history(prev_id);
				}
			}
		} else if (g_history.back().first == last_id.first && last_id.second > g_history.back().second + 1) {
			next_id.first = last_id.first;
			next_id.second = g_history.back().second + 1;
		}

		// Enqueue lines
		while (next_id <= last_id) {
			auto should_append_history = [&]() -> bool {
				if (g_mode != op_mode::rt_llama)
					return false;
				if (g_history.empty()) {
					if (next_id.second == 0)
						return true;
					return false;
				}
				auto next = g_lines.next(g_history.back());
				if (next == c_bad_id) {
					if (next_id.second)
						return false;
					return prev_id == g_history.back();
				}
				if (next == next_id) {
					return true;
				}
				return false;
			};
			if (std::unique_lock lock(g_mutex); should_append_history()) {
				// Update history
				if (next_id.second == 0 && !g_history.empty()) {
					// New segment started, remember previous segment name for restoring the history
					auto& prev = g_lines.segs[g_history.back().first].src_name;
					auto& list = g_lines.segs[next_id.first].prev_segs;
					list.emplace_back(prev);
					g_history.push_back(next_id);
					lock.unlock();
					if (!translate(params, {0, 0}, tr_cmd::sync))
						return 1;
					if (!update_segment(next_id.first, false))
						return 1;
					if (!g_lines[next_id].tr_text.empty()) {
						if (!translate(params, {next_id.first, 0}, tr_cmd::reload))
							return 1;
					}
				} else {
					g_history.push_back(next_id);
				}
			} else if (g_mode == op_mode::rt_llama && std::find(g_history.rbegin(), g_history.rend(), next_id) == g_history.rend()) {
				lock.unlock();
				// Stop worker thread first
				if (!translate(params, {0, 0}, tr_cmd::sync))
					return 1;
				if (g_history.back() == prev_id)
					update_segment(prev_id.first);
				get_incompletes(false);
				bool reload = true;
				if (incompletes.count(next_id.first)) {
					// Jump to another segment in progress (switching routes)
					prev_id = incompletes[next_id.first].first;
					next_id = std::min(next_id, incompletes[next_id.first].second);
					load_history(prev_id);
				} else if (g_lines[next_id].tr_text.empty()) {
					if (!g_lines[prev_id].tr_text.empty() && g_lines.is_last(prev_id)) {
						// Reload history from previous position of a completed segment
						next_id.second = 0;
						load_history(prev_id);
					} else {
						prev_id = c_bad_id;
					}
				} else {
					// Bruteforce.
					prev_id = c_bad_id;
					for (auto& [seg, iid] : incompletes) {
						load_history(iid.first);
						auto found = std::find(g_history.rbegin(), g_history.rend(), next_id);
						if (found != g_history.rend()) {
							// Just set valid prev_id, it will be overwritten later with next_id
							prev_id = iid.first;
							break;
						}
					}
				}
				if (prev_id == c_bad_id && g_lines.is_last(g_history.back()) && g_lines.segs[next_id.first].lines[0].tr_text.empty()) {
					// Start new segment from the start
					std::cerr << "Unexpected jump to the middle of segment." << std::endl;
					prev_id = g_history.back();
					next_id = last_id;
					next_id.second = 0;
					continue;
				}
				if (prev_id == c_bad_id) {
					std::cerr << "Unexpected jump, ignored. Try to use ^N to finish current segment." << std::endl;
					prev_id = g_history.back();
					next_id = g_lines.next(prev_id);
					break;
				}
				if (reload && !translate(params, c_bad_id, tr_cmd::reload))
					return 1;
			}

			id_queue.push_back(next_id);
			prev_id = next_id;
			if (!g_lines.advance(next_id))
				break;

			if (file_list.empty())
				continue;

			// Auto-continuation of repeated lines (compensate for repetition suppression)
			if (next_id > last_id && g_lines[next_id].sq_text == g_lines[prev_id].sq_text) {
				g_lines.advance(last_id);
				continue;
			}

			// List choices that are supposed to appear on the screen
			if (next_id > last_id && g_lines[next_id].name.starts_with("選択肢#")) {
				g_lines.advance(last_id);
				continue;
			}
		}

		if (!flush_id_queue())
			return 1;
	}

	update_input_time();
	if (g_mode == op_mode::rt_llama && !g_history.empty()) {
		// Stop background thread and discard
		if (!translate(params, g_lines.next(g_history.back()), tr_cmd::sync) || !translate(params, {}, tr_cmd::unload))
			return 1;
		// Save pending translations
		if (!update_segment(g_history.back().first))
			return 1;
	}

	// Print some stats
	if (g_stats->last_time.load()) {
		auto total_time = g_stats->last_time.load() - g_stats->start_time.load();
		auto read_time0 = g_stats->rt_reading_ms.load() / 1000;
		auto afk_time = std::accumulate(std::begin(g_stats->rt_afk_ms), std::end(g_stats->rt_afk_ms), ui64(0)) / 1000;
		time_t start = g_stats->start_time.load();
		std::cerr << "Started: " << std::put_time(std::localtime(&start), "%Y-%m-%d %X") << std::endl;
		std::cerr << "Reading time: " << read_time0 / 36 / 100. << "h (";
		std::cerr << 10000 * read_time0 / (1 + read_time0 + afk_time) / 100. << "%)" << std::endl;
		std::cerr << "AFK time: " << afk_time / 36 / 100. << "h" << std::endl;
		// Time spent in sampling logic, but count is the number of resulting tokens.
		if (auto count = g_stats->raw_samples.load()) {
			std::cerr << "Sample speed: ";
			std::cerr << uint(10 / (g_stats->sample_time.load() / count / 1e6)) / 10. << "/s (raw: ";
			std::cerr << count << ")" << std::endl;
		}
		// Time spent in batching big amount of tokens (including prompt).
		if (auto count = g_stats->batch_count.load()) {
			std::cerr << "Batch speed: ";
			std::cerr << uint(10 / (g_stats->batch_time.load() / count / 1e6)) / 10. << "/s (total: ";
			std::cerr << count << ")" << std::endl;
		}
		// Time spent in batching few tokens, no separate counter for eval (uses sample counter).
		if (auto count = g_stats->sample_count.load()) {
			std::cerr << "Eval speed: ";
			std::cerr << uint(10 / (g_stats->eval_time.load() / count / 1e6)) / 10. << "/s (total: ";
			std::cerr << count << ")" << std::endl;
		}
		// Time spent in preparation for eval
		if (auto count = g_stats->lag_count.load()) {
			std::cerr << "Eval delay: ~";
			std::cerr << g_stats->lag_time.load() / count / 1000 / 1000. << "s (total: " << count << ")" << std::endl;
		}
		ui64 tr_count = 0;
		for (auto& line : g_lines) {
			if (!line.tr_text.empty())
				tr_count++;
		}
		if (tr_count) {
			std::cerr << "Translated: " << tr_count << "/" << g_lines.count_lines();
			std::cerr << " (" << 100 * read_time0 / tr_count / 100. << "s/line)" << std::endl;
		}
		if (auto count = g_stats->raw_translates.load()) {
			std::cerr << "Attempts: " << count;
			auto acpt = g_stats->raw_accepts.load() * 10000 / count / 100.;
			auto disc = g_stats->raw_discards.load() * 10000 / count / 100.;
			std::cerr << " (accept " << acpt << "%; discard " << disc << "%)" << std::endl;
		}
		if (total_time && tr_count) {
			time_t end = g_stats->start_time.load() + total_time * 1000 / tr_count * g_lines.count_lines() / 1000;
			std::cerr << "Estimated finish: " << std::put_time(std::localtime(&end), "%Y-%m-%d") << std::endl;
		}
	}

	if (g_stop)
		std::cerr << "Terminated." << std::endl;

	return 0;
}

static_assert(natural_order_less()("v-1", "v-2"));
static_assert(natural_order_less()("v1", "v01"));
static_assert(natural_order_less()("v01", "v2"));
static_assert(natural_order_less()("v01", "v10"));
static_assert(natural_order_less()("v2", "v10"));
static_assert(natural_order_less()("v1", "v") == false);
static_assert(natural_order_less()("a", "a0"));
static_assert(natural_order_less()("a0", "b"));
static_assert(natural_order_less()("abcde", "99999") == false);
static_assert(natural_order_less()("99999", "abcde"));
