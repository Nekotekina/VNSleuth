#include "main.hpp"
#include "common.h"
#include "llama.h"
#include "parser.hpp"
#include <algorithm>
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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

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

// Translation example injected after prompt
std::string example = "JP: この物語はフィクションです。\n"
					  "En: This story is a work of fiction.\n";

// Print line, return original speaker name if first encountered
std::string print_line(line_id id, std::string* line = nullptr, bool stream = false)
{
	std::string out = apply_replaces(g_lines[id].text);
	// TODO: implement as regex replace
	if (out.starts_with("…") && out != "…")
		out.erase(0, "…"sv.size());

	std::string speaker;
	if (g_lines[id].name.starts_with("選択肢#")) {
		// Insert translated choice selection prefix
		speaker += " Choice ";
		speaker += g_lines[id].name.substr("選択肢#"sv.size());
	} else if (!g_lines[id].name.empty()) {
		// Find registered speaker translation
		std::lock_guard lock(g_mutex);

		const auto& found = g_speakers.at(g_lines[id].name);
		if (!found.empty()) {
			speaker += " ";
			speaker += found;
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
		line->append(isuffix);
	}

	return speaker;
}

std::string squeeze_line(const std::string& line)
{
	// For opportunistic compatibility with "bad" hooks which repeat characters, or remove repeats
	std::string r;
	std::string_view prev, next;
	for (const char& c : line) {
		next = {};
		// Decode UTF-8 sequence (TODO: better validation?)
		if ((c & 0x80) == 0) {
			next = std::string_view(&c, 1);
		} else if ((c & 0xe0) == 0xc0) {
			next = std::string_view(&c, 2);
		} else if ((c & 0xf0) == 0xe0) {
			next = std::string_view(&c, 3);
		} else if ((c & 0xf8) == 0xf0) {
			next = std::string_view(&c, 4);
		}
		for (auto& ch : next) {
			// Check for null terminator (`next` could be out of bounds)
			if (ch == 0)
				return r;
			if (&ch > next.data() && (ch & 0xc0) != 0x80) {
				// Invalid UTF-8 sequence
				next = {};
				prev = {};
				break;
			}
		}
		if (!next.empty() && next != prev) {
			// Skip spaces (both ASCII and full-width) and control characters
			if (next == "　"sv || next == " "sv)
				continue;
			if (next[0] + 0u < 32 || next[0] == '\x7f') {
				prev = {};
				continue;
			}
			// Append non-repeating code
			r += next;
			prev = next;
		}
	}
	return r;
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
		for (auto& [orig, tr] : g_speakers) {
			if (tr != ":")
				names << orig << tr << "\n";
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

bool update_segment(uint seg, bool upd_names = true)
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
		if (line.tr_text.empty())
			break;
		dump += line.tr_text;
	}
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
					g_lines[id].seed = 0;
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
		g_lines[id2].seed = 0;
	}

	// Check and load what remains in the file
	auto old = std::move(g_lines.segs[seg].tr_tail);
	g_lines.segs[seg].tr_tail = std::move(text);
	if (keep && g_lines.segs[seg].tr_tail == old)
		kept = -1;

	return std::make_pair(id.second, verify ? !is_broken : kept);
}

void sigh(int) { g_stop = true; }

int main(int argc, char* argv[])
{
	g_mode = op_mode::rt_llama;
	if (argc == 2) {
		g_mode = op_mode::rt_cached;
	} else if (argc >= 3 && argv[2] == "--legacy"sv) {
		g_mode = op_mode::print_only;
	} else if (argc >= 3 && argv[2] == "--check"sv) {
		g_mode = op_mode::print_info;
	} else if (argc < 2) {
		std::cerr << "VNSleuth v0.3" << std::endl;
		std::cerr << "Usage 0: " << argv[0] << " <game_directory> --color" << std::endl;
		std::cerr << "\tUse existing translation cache only (only useful if it's complete)." << std::endl;
		std::cerr << "Usage 1: " << argv[0] << " <game_directory> --check" << std::endl;
		std::cerr << "\tPrint furigana, some information, initialize translation cache." << std::endl;
		std::cerr << "Usage 2: " << argv[0] << " <game_directory> --color -m <model> [<LLAMA args>...]" << std::endl;
		std::cerr << "\tStart translating in real time (recommended)." << std::endl;
		std::cerr << "Nothing to do." << std::endl;
		return 1;
	}

	if (argv[1] == "--help"sv) {
		std::cerr << "(TODO) run without arguments to see usage example." << std::endl;
		return 1;
	}

	gpt_params params{};
	params.n_gpu_layers = 999;
	//params.cache_type_k = "q8_0";
	//params.cache_type_v = "q8_0";
	params.seed = 0;
	params.n_ctx = 4096;
	params.n_batch = 2048;
	params.n_predict = 128;
	params.sparams.temp = 1;
	params.sparams.temp = 0.2;
	params.sparams.top_p = 0.3;
	params.sparams.penalty_last_n = 3;
	params.sparams.penalty_repeat = 1.1;
	params.model = ".";

	// Basic param check
	std::string dir_name = argv[1];
	if (argc > 2) {
		argc--;
		argv++;
		if (g_mode == op_mode::print_only || g_mode == op_mode::print_info) {
			argc--;
			argv++;
		}
		if (!gpt_params_parse(argc, argv, params)) {
			gpt_params_print_usage(argc, argv, params);
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
		if (g_mode == op_mode::rt_llama && params.model == ".")
			g_mode = op_mode::rt_cached;
	}

	// Parse scripts in a given directory
	std::string line, prompt_path;
	bool is_incomplete = false;
	bool is_broken = false;

	// Load file list recursively
	vnsleuth_path = fs::absolute(fs::path(dir_name)).lexically_normal() / "__vnsleuth";
	std::map<std::string, std::size_t, natural_order_less> file_list;
	if (fs::is_directory(dir_name)) {
		fs::create_directory(vnsleuth_path);
		fs::path path = vnsleuth_path;
		replaces_path = path / "__vnsleuth_replace.txt";
		prompt_path = path / "__vnsleuth_prompt.txt";
		names_path = path / "__vnsleuth_names.txt";
		for (const auto& entry : fs::recursive_directory_iterator(path.parent_path(), fs::directory_options::follow_directory_symlink)) {
			if (entry.is_regular_file()) {
				// Skip special files
				const auto fname = entry.path().filename().string();
				if (fname.starts_with("__vnsleuth_") && fname.ends_with(".txt"))
					continue;
				// Check file size
				const auto size = entry.file_size();
				if (size > 10)
					file_list.emplace(entry.path(), size);
			}
		}
	} else {
		std::cerr << "Not a directory: " << dir_name << std::endl;
		return 1;
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

	auto reload_names = [&] {
		// Load (pre-translated) names
		std::lock_guard lock(g_mutex);

		std::ifstream names(names_path);
		if (names.is_open()) {
			g_speakers.clear();
			while (std::getline(names, line, '\n')) {
				if (line.ends_with("\r"))
					line.erase(line.end() - 1);
				if (!line.ends_with(":")) {
					// Translated name is either empty or ends with another ':'
					std::cerr << "Failed to parse translated name string: " << line << std::endl;
					continue;
				}
				if (const auto pos = line.find_first_of(":") + 1)
					g_speakers.emplace(line.substr(0, pos), line.substr(pos));
			}
		} else {
			std::cerr << "Translation names not found: " << names_path << std::endl;
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
					if (line.ends_with("\r"))
						line.erase(line.end() - 1);
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
	std::cerr << "Loaded names: " << g_speakers.size() - 1 << std::endl;
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
		if (!g_lines.segs.empty() && g_speakers.size() > 1) {
			update_names(names_path);
		}

		if (argc > 3 && argv[3] == "--repair"sv) {
			// Update all segments
			for (auto& seg : g_lines.segs) {
				if (!seg.lines.at(0).tr_text.empty() || !seg.tr_tail.empty()) {
					update_segment(&seg - g_lines.segs.data(), false);
				}
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
					// Inbetween segments
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

	if (g_mode == op_mode::rt_llama) {
		load_history(prev_id);
		// Also kick translator
		if (prev_id != c_bad_id) {
			if (!translate(params, prev_id, tr_cmd::kick))
				return false;
		}
	}

	// List of IDs that must be processed (possibly more than once)
	std::vector<line_id> id_queue;
	auto flush_id_queue = [&](bool rewrite = false) -> bool {
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

		bool is_kicked = false;
		for (auto& id : id_queue) {
			if (rewrite) {
				if (!translate(params, id))
					return false;
				continue;
			}
			std::unique_lock lock(g_mutex);
			auto& line = g_lines[id];
			if (!line.tr_text.empty()) {
				auto out = line.tr_text; // copy

				static const auto real_iprefix = iprefix + (iprefix.ends_with(" ") ? "" : " ");
				static const auto alt_iprefix = "\n" + real_iprefix;
				if (!out.starts_with(real_iprefix)) {
					if (auto pref_pos = out.find(alt_iprefix) + 1) {
						out.insert(0, g_esc.orig);
						out.erase(pref_pos, alt_iprefix.size() - 1);
					} else {
						std::cerr << iprefix << " not found in translation cache." << std::endl;
						return false;
					}
				} else {
					out.replace(0, iprefix.size(), g_esc.orig);
				}

				static const auto real_isuffix = "\n" + isuffix + (isuffix.ends_with(" ") ? "" : " ");
				const auto suff_pos = out.find(real_isuffix);
				if (suff_pos + 1 == 0) {
					std::cerr << isuffix << " not found in translation cache." << std::endl;
					return false;
				}
				out.replace(suff_pos + 1, real_isuffix.size() - 1, g_esc.tran);
				std::cout << out << g_esc.reset << std::flush;
				lock.unlock();

				if (g_mode == op_mode::rt_llama && !is_kicked) {
					is_kicked = true;
					if (!translate(params, id, tr_cmd::kick))
						return false;
				}
				if (g_mode == op_mode::rt_llama && g_lines.is_last(id)) {
					// Prepare to autosave segment on last id
					if (!translate(params, line_id{id.first, 0u - 1}, tr_cmd::sync))
						return false;
				}
			} else if (g_mode == op_mode::rt_cached) {
				std::cerr << "Error: Translation cache is incomplete" << std::endl;
				break;
			} else {
				lock.unlock();
				if (!translate(params, id))
					return false;
			}
		}

		if (g_mode != op_mode::rt_llama) {
			id_queue.clear();
		}
		return true;
	};

	std::string last_line;
	if (g_mode == op_mode::rt_llama) {
		// Show previous message(s) in a limited fashion (one message + selections)
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
			flush_id_queue();
		}
	}

	// Setup Ctrl+C handler
	signal(SIGINT, sigh);

	// When line is found and is expected, it's printed out (back_buf remains empty).
	// When line is found and is unique, the back buffer is flushed as well.
	// When line is found but not unique, it's added to the back buffer.
	// Previous non-unique lines are printed if they exactly precede unique line, otherwise wiped.
	std::vector<std::string> back_buf;
	while (!g_stop && std::getline(std::cin, line)) {
		if (g_mode == op_mode::rt_llama) {
			if (line.empty()) {
				// Process rewrite request
				if (id_queue.empty() || id_queue.size() > 10 || id_queue.back() != g_history.back()) {
					std::cerr << "Cannot process rewrite request here." << std::endl;
					continue;
				}
				for (auto& id : id_queue) {
					g_lines[id].seed++;
				}
				if (!flush_id_queue(true))
					return 1;
				continue;
			}

			if (line == "\03") {
				// Process terminate request (^C)
				g_stop = true;
				break;
			}

			if (line == "\x12") {
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
				if (!flush_id_queue(false))
					return 1;
				continue;
			}

			if (line == "\x13") {
				// Process stop-and-save request (^S)
				if (!translate(params, next_id, tr_cmd::sync))
					return 1;
				if (prev_id != c_bad_id && update_segment(prev_id.first)) {
					std::cerr << "Wrote file: " << g_lines.segs[prev_id.first].cache_name << std::endl;
				}
				continue;
			}
		}

		std::string sq_line = squeeze_line(line);
		if (sq_line.empty() || sq_line == last_line) {
			continue;
		}
		if (next_id != c_bad_id) {
			// Partial match of predicted next line, should be at least 1/3 of original length
			const auto& sq = g_lines[next_id].sq_text;
			if (sq_line.size() >= sq.size() / 3 && sq.find(sq_line) + 1) {
				sq_line = sq;
			}
		}
		auto it = g_strings.find(sq_line);
		if (next_id == c_bad_id && (it == g_strings.end() || it->second == c_bad_id) && back_buf.empty()) {
			// For smoother startup from an ambiguous line, attempt to show first line in a segment
			const auto it2 = g_start_strings.find(sq_line);
			if (it2 != g_start_strings.end()) {
				next_id = it2->second;
			} else {
				for (const auto& [sq, id] : g_start_strings) {
					// Partial match fallback
					if (sq_line.size() >= sq.size() / 3 && sq.find(sq_line) + 1) {
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
		if (it == g_strings.end()) {
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
			if (it->second == c_bad_id) {
				// Remember ambiguous line
				std::cerr << g_esc.buf << "Buffered: " << line << std::endl;
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
			if (next_id != c_bad_id)
				next_id.second = 0;
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
				if (prev_id == c_bad_id) {
					std::cerr << "Unexpected jump, ignored. Could be a bug, or try to restart." << std::endl;
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

	if (g_mode == op_mode::rt_llama && !g_history.empty()) {
		// Stop background thread and discard
		if (!translate(params, g_lines.next(g_history.back()), tr_cmd::sync))
			return 1;
		// Save pending translations
		if (!update_segment(g_history.back().first))
			return 1;
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
