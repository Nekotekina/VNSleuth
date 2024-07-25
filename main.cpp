#include "main.hpp"
#include "common.h"
#include "llama.h"
#include "tools/tiny_sha1.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
	// Reduce token count and also fix potential issues with borked translations
	std::string out;
	out += g_lines[id].text;
	REPLACE(out, "\t", "　");
	REPLACE(out, "〜", "～"); // Use one more typical for SJIS-WIN
	REPLACE(out, "……", "…");
	REPLACE(out, "──", "─");
	REPLACE(out, "――", "―");
	REPLACE(out, "ーーー", "ーー"); // Reduce repetitions
	REPLACE(out, "ーー", "～");		// Replace "too long" sound with ～
	REPLACE(out, "～～", "～");
	REPLACE(out, "「…", "「"); // Sentences starting with … might get translated as empty "..."
	REPLACE(out, "（…", "（");
	REPLACE(out, "『…", "『");
	if (out.starts_with("…") && out != "…")
		out.erase(0, "…"sv.size());
	REPLACE(out, "？？", "？");
	REPLACE(out, "ぁぁ", "ぁ");
	REPLACE(out, "ぃぃ", "ぃ");
	REPLACE(out, "ぅぅ", "ぅ");
	REPLACE(out, "ぇぇ", "ぇ");
	REPLACE(out, "ぉぉ", "ぉ");
	REPLACE(out, "ゃゃ", "ゃ");
	REPLACE(out, "ゅゅ", "ゅ");
	REPLACE(out, "ょょ", "ょ");
	REPLACE(out, "っっ", "っ");
	REPLACE(out, "ァァ", "ァ");
	REPLACE(out, "ィィ", "ィ");
	REPLACE(out, "ゥゥ", "ゥ");
	REPLACE(out, "ェェ", "ェ");
	REPLACE(out, "ォォ", "ォ");
	REPLACE(out, "ャャ", "ャ");
	REPLACE(out, "ュュ", "ュ");
	REPLACE(out, "ョョ", "ョ");
	REPLACE(out, "ッッ", "ッ");

	auto result = g_lines[id].name;
	bool ret = false;

	std::string speaker;
	if (result.starts_with("選択肢#")) {
		// Insert translated choice selection prefix
		speaker += " Choice ";
		speaker += result.substr("選択肢#"sv.size());
	} else if (!result.empty()) {
		// Find registered speaker translation
		std::lock_guard lock(g_mutex);

		const auto& found = g_speakers.at(result);
		if (found.empty()) {
			ret = true;
		} else if (found != ":") {
			speaker += " ";
			speaker += found;
		}
	}

	if (g_mode == op_mode::print_only) {
		// Passthrough mode (print-only): add \ for multiline input
		std::cout << iprefix << result << out << "\\" << std::endl;
		// Add Ctrl+D for same-line output
		std::cout << isuffix << speaker << "\04" << std::flush;
	}

	if (line) {
		line->append(iprefix);
		line->append(result);
		line->append(out);
		line->append("\n");
		line->append(isuffix);
		line->append(speaker);
	}

	if (stream) {
		// Print colored output without prefixes
		if (speaker.starts_with(" "))
			speaker.erase(0, 1);
		std::cout << "\033[0;33m" << result << out << std::endl;
		std::cout << "\033[0;1m" << speaker << std::flush;
	}

	if (ret) {
		return result;
	}
	return "";
}

namespace fs = std::filesystem;

static std::string names_path;

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

bool update_segment(std::string_view prompt, uint seg)
{
	std::lock_guard lock(g_mutex);

	update_names(names_path);

	std::string dump = "SRC:";
	dump += g_lines.segs[seg].src_name;
	dump += "\n";
	dump += prompt;
	for (auto& line : g_lines.segs[seg].lines) {
		if (line.tr_text.empty())
			break;
		dump += line.tr_text;
	}
	std::ofstream file(g_lines.segs[seg].cache_name + ".tmp", std::ios_base::trunc | std::ios_base::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open " << g_lines.segs[seg].cache_name << ".tmp" << std::endl;
		return false;
	}
	file.write(dump.data(), dump.size());
	file.close();
	fs::rename(g_lines.segs[seg].cache_name + ".tmp", g_lines.segs[seg].cache_name);
	return true;
}

uint load_translation(uint seg, const std::string& path)
{
	std::ifstream cache(path);
	if (!cache.is_open())
		throw std::runtime_error("Failed to open " + path);

	// Extract cached translation
	line_id id{seg, 0};
	std::string temp;
	while (std::getline(cache, temp, '\n')) {
		// Skip prompt+example (example isn't stored anymore, it's an artifact)
		if (temp.starts_with(iprefix) && !temp.ends_with("\\")) {
			std::string text = std::move(temp);
			if (std::getline(cache, temp, '\n')) {
				// Store both original and translated lines as a single string
				text += '\n';
				text += temp;
				text += '\n';
				g_lines[id].tr_text = std::move(text);
				g_lines[id].seed = 0;
				g_lines.advance(id);
			} else {
				break;
			}
		}
	}

	// Clear remaining lines
	for (auto id2 = id; id2 != c_bad_id; g_lines.advance(id2)) {
		g_lines[id2].tr_text = {};
		g_lines[id2].seed = 0;
	}

	return id.second;
}

void sigh(int) { g_stop = true; }

int main(int argc, char* argv[])
{
	const bool is_piped = !isatty(STDIN_FILENO);
	g_mode = op_mode::print_only;
	if (argc == 2) {
		g_mode = is_piped ? op_mode::rt_cached : op_mode::print_info;
	} else if (argc > 2 && argv[2] != "-"sv) {
		g_mode = is_piped ? op_mode::rt_llama : op_mode::make_cache;
	} else if (!is_piped) {
		argc = 1;
	}

	if (argc < 2) {
		std::cerr << "VNSleuth v0.2" << std::endl;
		std::cerr << "Usage: " << argv[0] << " <script_directory> [-m <model> <LLAMA args>...]" << std::endl;
		std::cerr << "TTY+no args = print furigana, some information, and exit." << std::endl;
		std::cerr << "TTY+model = generate translation cache." << std::endl;
		std::cerr << "pipe+- = only send preprocessed lines to stdout." << std::endl;
		std::cerr << "pipe+no args = print cached translation in real time (must be created)." << std::endl;
		std::cerr << "pipe+model = translate in real time (recommended)." << std::endl;
		std::cerr << "Nothing to do." << std::endl;
		return 1;
	}

	if (argv[1] == "--help"sv) {
		std::cerr << "(TODO) run without arguments to see usage example." << std::endl;
		return 1;
	}

	// Get startup timestamp
	const auto g_now = std::chrono::steady_clock::now();

	// Parse scripts in a given directory
	std::string line, cache_path, prompt_path;
	bool is_incomplete = false;

	// Load file list recursively
	std::string dir_name = argv[1];
	std::vector<std::pair<std::string, std::size_t>> file_list;
	if (fs::is_regular_file(dir_name)) {
		fs::path path = fs::absolute(fs::path(dir_name)).lexically_normal();
		while (true) {
			fs::path parent = path.parent_path();
			prompt_path = parent / "__vnsleuth_prompt.txt";
			names_path = parent / "__vnsleuth_names.txt";
			if (parent == path.root_path()) {
				prompt_path.clear();
				names_path.clear();
				break;
			}
			if (fs::is_regular_file(prompt_path) || fs::is_regular_file(names_path)) {
				break;
			}
			path = std::move(parent);
		}
		file_list.emplace_back(std::move(dir_name), fs::file_size(dir_name));
		if (g_mode == op_mode::make_cache)
			is_incomplete = true;
	} else {
		fs::path path = fs::absolute(fs::path(dir_name)).lexically_normal();
		prompt_path = path / "__vnsleuth_prompt.txt";
		names_path = path / "__vnsleuth_names.txt";
		for (const auto& entry : fs::recursive_directory_iterator(path, fs::directory_options::follow_directory_symlink)) {
			if (entry.is_regular_file()) {
				// Skip special files
				const auto fname = entry.path().filename().string();
				if (fname.starts_with("__vnsleuth_") && fname.ends_with(".txt"))
					continue;
				// Check file size (must be less than 4 GiB)
				const auto size = entry.file_size();
				if (size > 10 && size < 4ull * 1024 * 1024 * 1024)
					file_list.emplace_back(entry.path(), size);
			}
		}
	}

	if (prompt_path.empty() || !fs::is_regular_file(prompt_path)) {
		std::cerr << "Translation prompt not found: " << prompt_path << std::endl;
		if (g_mode == op_mode::make_cache || g_mode == op_mode::rt_llama) {
			std::cerr << "Translation prompt is required." << std::endl;
			return 1;
		}

		if (!prompt_path.empty() && g_mode == op_mode::print_info) {
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

	// Sort file list alphabetically
	std::sort(file_list.begin(), file_list.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

	for (uint i = 0; i < file_list.size(); i++) {
		const int fd = open(file_list[i].first.c_str(), O_RDONLY);
		if (fd >= 0) {
			// Map file in memory (alloc 1 more zero byte, like std::string)
			const auto map_sz = file_list[i].second + 1;
			const std::shared_ptr<void> file(mmap(0, map_sz, PROT_READ, MAP_PRIVATE, fd, 0), [=](void* ptr) {
				munmap(ptr, map_sz);
				close(fd);
			});

			// View file as chars
			const std::string_view data(static_cast<char*>(file.get()), file_list[i].second);

			{
				// Generate SHA1 hash (only read first 1 MiB), generate cache path
				char buf[42]{};
				sha1::SHA1 s;
				s.processBytes(data.data(), std::min<std::size_t>(1024 * 1024, data.size()));
				std::uint32_t digest[5];
				s.getDigest(digest);
				std::snprintf(buf, 41, "%08x%08x%08x%08x%08x", digest[0], digest[1], digest[2], digest[3], digest[4]);
				cache_path = fs::path(file_list[i].first).parent_path();
				cache_path += "/__vnsleuth_";
				cache_path += buf;
				cache_path += ".txt";
			}

			g_lines.segs.emplace_back();
			g_lines.segs.back().cache_name = cache_path;
			g_lines.segs.back().src_name = file_list[i].first;
			if (parse(data)) {
				if (g_mode == op_mode::print_info || g_mode == op_mode::rt_cached || g_mode == op_mode::rt_llama) {
					if (g_mode == op_mode::print_info) {
						std::cerr << "Found file: " << file_list[i].first << std::endl;
						std::cerr << "Cache file: " << cache_path;
					}
					if (fs::is_regular_file(cache_path)) {
						const auto tr_lines = load_translation(i, cache_path);
						if (tr_lines < g_lines.segs.at(i).lines.size() && g_mode == op_mode::print_info)
							std::cerr << " (partial)" << std::endl;
						else if (g_mode == op_mode::print_info)
							std::cerr << (tr_lines ? " (exists)" : " (empty!)") << std::endl;
					} else {
						if (g_mode == op_mode::print_info)
							std::cerr << " (not found)" << std::endl;
						is_incomplete = true;
					}
				}
			} else {
				g_lines.segs.pop_back();
			}
		} else if (file_list.size() == 1) {
			std::cerr << "Error: Could not open file: " << file_list[i].first << std::endl;
			return 1;
		}
	}

	if (g_mode != op_mode::make_cache || !is_incomplete) {
		std::cerr << "Loaded files: " << g_lines.segs.size() << std::endl;
		std::cerr << "Loaded lines: " << g_lines.count_lines() << std::endl;
		std::cerr << "Loaded names: " << g_speakers.size() - 1 << std::endl;
	}

	if (is_incomplete) {
		if (g_mode == op_mode::print_info || g_mode == op_mode::rt_cached)
			std::cerr << "Some cache files were not found." << std::endl;
	}

	if (g_mode == op_mode::print_info || g_lines.segs.empty()) {
		// Print known furigana
		for (auto&& [type_as, read_as] : g_furigana) {
			std::cerr << type_as << "=" << read_as << std::endl;
		}

		// Update names
		if (!g_lines.segs.empty() && g_speakers.size() > 1) {
			update_names(names_path);
		}

		std::cerr << "Nothing to do." << std::endl;
		return 0;
	}

	gpt_params params{};
	params.n_gpu_layers = 999;
	params.seed = 0;
	params.n_ctx = 4096;
	params.n_predict = 128;
	params.sparams.temp = 0.2;
	params.sparams.top_p = 0.3;
	params.sparams.penalty_last_n = 3;
	params.sparams.penalty_repeat = 1.1;
	auto reload_prompt = [&]() {
		// Load prompt
		std::lock_guard lock(g_mutex);

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
		}
	};
	reload_prompt();
	if (argc > 2) {
		if (!gpt_params_parse(argc - 1, argv + 1, params)) {
			gpt_params_print_usage(argc - 1, argv + 1, params);
			return 1;
		}

		// Initialize llama.cpp
		params.n_batch = params.n_ctx;
		if (!translate(params, c_bad_id))
			return 1;
	}

	if (is_piped) {
		std::cerr << "Waiting for input..." << std::endl;
		// Hack to prevent interleaving stderr/stdout
		usleep(300'000);
	}

	line_id next_id = c_bad_id;
	line_id prev_id = c_bad_id;

	// List of IDs that must be processed (possibly more than once)
	std::vector<line_id> id_queue;
	auto flush_id_queue = [&](bool rewrite = false) -> bool {
		if (g_mode == op_mode::print_only) {
			for (auto& id : id_queue) {
				if (static bool prompt_sent = false; !std::exchange(prompt_sent, true)) {
					// Send prompt and example before first line
					std::string out = params.prompt + example;
					REPLACE(out, "\n", "\\\n");
					std::cout << out << std::flush;
				}
				print_line(id);
			}
			id_queue.clear();
			return true;
		}

		bool is_kicked = false;
		for (auto& id : id_queue) {
			if (g_mode != op_mode::make_cache && id.second == 0 && !rewrite) {
				std::cerr << "Starting new segment..." << std::endl << std::flush;
			}
			if (rewrite) {
				if (!translate(params, id))
					return false;
				continue;
			}
			std::unique_lock lock(g_mutex);
			auto& line = g_lines[id];
			if (!line.tr_text.empty()) {
				if (g_mode == op_mode::make_cache)
					continue;
				auto out = line.tr_text; // copy
				if (!out.starts_with(iprefix)) {
					std::cerr << iprefix << " not found in translation cache." << std::endl;
					return false;
				}
				out.replace(0, iprefix.size(), "\033[0;33m");
				static const auto real_isuffix = "\n" + isuffix + (isuffix.ends_with(" ") ? "" : " ");
				const auto suff_pos = out.find(real_isuffix);
				if (suff_pos + 1 == 0) {
					std::cerr << isuffix << " not found in translation cache." << std::endl;
					return false;
				}
				out.replace(suff_pos + 1, real_isuffix.size() - 1, "\033[0;1m");
				std::cout << out << "\033[0m" << std::flush;
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

	// Setup Ctrl+C handler
	signal(SIGINT, sigh);

	if (g_mode == op_mode::make_cache) {
		// Demand all possible IDs
		for (uint i = 0; !g_stop && i < g_lines.segs.size(); i++) {
			const auto _now = std::chrono::steady_clock::now();
			const auto nlines = g_lines.segs[i].lines.size();
			std::cerr << "Translating: " << g_lines.segs[i].src_name << " -> " << g_lines.segs[i].cache_name << std::endl;
			std::cerr << "Translated lines: 0/" << nlines << std::flush;
			for (uint j = 0; j < nlines; j++) {
				id_queue.emplace_back(i, j);
				if (!flush_id_queue())
					return 1;
				const auto stamp = std::chrono::steady_clock::now();
				const auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stamp - _now);
				std::cerr << "\rTranslated lines: " << (j + 1) << "/" << nlines;
				std::cerr << "... (" << elaps.count() / (j + 1) / 1000. << "s/line)    ";
				std::cerr << std::flush;
				if (g_stop)
					break;
			}
			if (!translate(params, c_bad_id))
				return 1;
			const auto elaps = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - _now);
			std::cerr << std::endl << "Elapsed: " << elaps.count() << "s" << std::endl;
		}

		if (g_stop)
			std::cerr << "Aborted by user." << std::endl;
		const auto elaps = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - g_now);
		std::cerr << "Cache generation completed. Elapsed: ";
		std::cerr << (elaps.count() / 36) / 100. << " hours (" << elaps.count() << "s)" << std::endl;
		return 0;
	}

	// Exact match mode:
	// When line is found and is expected, it's printed out (back_buf remains empty).
	// When line is found and is unique, the back buffer is flushed as well.
	// When line is found but not unique, it's added to the back buffer.
	// Previous non-unique lines are printed if they exactly precede unique line, otherwise wiped.
	std::vector<std::string> back_buf;
	while (!g_stop && is_piped && std::getline(std::cin, line)) {
		if (g_mode == op_mode::rt_llama) {
			if (line.empty()) {
				// Process rewrite request
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
				reload_prompt();
				for (auto& seg : g_lines.segs) {
					std::lock_guard lock{g_mutex};

					if (!seg.lines[0].tr_text.empty()) {
						if (!load_translation(&seg - g_lines.segs.data(), seg.cache_name)) {
							throw std::runtime_error("Broken cache " + seg.cache_name);
						}
					}
				}

				if (!translate(params, c_bad_id))
					return 1;
				std::cerr << "Cache reloaded." << std::endl << std::flush;
				if (!flush_id_queue(false))
					return 1;
				continue;
			}

			if (line == "\x13") {
				// Process stop-and-save request (^S)
				if (!translate(params, next_id, tr_cmd::sync))
					return 1;
				if (prev_id != c_bad_id && update_segment(params.prompt, prev_id.first)) {
					std::cerr << "Wrote file: " << g_lines.segs[prev_id.first].cache_name << std::endl;
				}
				continue;
			}
		}

		// TODO: filter duplicates (currently filtered by xclipmonitor)
		if (line.size() && next_id != c_bad_id && g_lines[next_id].text.find(line) + 1) {
			// Partial match of predicted next line
			line = g_lines[next_id].text;
		}
		const auto it = g_strings.find(line);
		if (line.empty() || it == g_strings.end()) {
			// Line not found (garbage?)
			//std::cerr << "Ignored line: " << line << std::endl;
			continue;
		}
		if (g_mode == op_mode::rt_llama) {
			id_queue.clear();
		}
		line_id last_id = next_id;
		if (last_id == c_bad_id || g_lines[last_id].text != line) {
			// Handle unexpected line:
			if (it->second == c_bad_id) {
				// Remember ambiguous line
				back_buf.emplace_back(std::move(line));
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
					if (g_lines[next_id].text == *itr) {
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

		// Enqueue lines
		while (next_id <= last_id) {
			id_queue.push_back(next_id);
			prev_id = next_id;
			if (!g_lines.advance(next_id))
				break;

			// Auto-continuation of repeated lines (compensate for repetition suppression)
			if (next_id > last_id && g_lines[next_id].text == g_lines[prev_id].text) {
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

	if (g_stop)
		std::cerr << "Terminated by user." << std::endl;

	if (g_mode == op_mode::rt_llama) {
		// Flush (save pending translations)
		if (!translate(params, next_id, tr_cmd::sync))
			return 1;
		if (prev_id != c_bad_id && !update_segment(params.prompt, prev_id.first))
			return 1;
	}

	return 0;
}
