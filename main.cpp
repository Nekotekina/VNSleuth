#include "main.hpp"
#include "tools/tiny_sha1.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>
#endif

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

// Guiding prefix for original lines of text
std::string iprefix = "JP: ";

// En: prefix doesn't contain a space, for an interesting reason.
// For Chinese models it may turn into an explicit space token.
// In English text such token usually doesn't appear.
// However, it may appear in Chinese texts as a delimiter.
// This may provoke Chinese-speaking model to output Chinese.
std::string isuffix = "En:";

// Translation example injected after prompt (may be superfluous for some models or even harmful)
std::string_view example = "？？？:「猫か…」\\\n"
						   "En: ???: \"Ah, so it's a cat...\"\\\n"
						   "JP: この物語はフィクションです。\\\n"
						   "En: This story is a work of fiction.\\\n"
						   "JP: "sv;

// Print line, return original speaker name if first encountered
std::string print_line(size_t id, int fd = -1, std::string* line = nullptr, std::string_view prefix = ""sv)
{
	// Reduce token count and also fix potential issues with borked translations
	std::string out;
	out += g_text[id].second;
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

	auto result = std::string(g_text[id].first);
	bool ret = false;

	std::string speaker;
	if (result.starts_with("選択肢#")) {
		// Insert translated choice selection prefix
		speaker += " Choice ";
		speaker += result.substr("選択肢#"sv.size());
	} else if (!result.empty()) {
		// Find registered speaker translation
		const auto& found = g_speakers.at(result);
		if (found.empty()) {
			ret = true;
		} else if (found != ":") {
			speaker += " ";
			speaker += found;
		}
	}

	if (fd >= 0) {
		std::string buf(prefix);
		buf += result;
		buf += out;
		buf += "\\\n"; // Add \ for multiline input
		buf += isuffix;
		buf += speaker;
		buf += "\04\n"; // Finish input without a newline
		if (write(fd, buf.data(), buf.size()) + size_t{} != buf.size()) {
			perror("Writing to pipe failed");
			std::exit(1);
		}
	} else {
		std::cout << prefix << result << out << std::endl;
	}

	if (line) {
		line->append(prefix);
		line->append(result);
		line->append(out);
		line->append("\n");
		line->append(isuffix);
		line->append(speaker);
	}

	if (ret) {
		return result;
	}
	return "";
}

namespace fs = std::filesystem;

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
	} else {
		std::cerr << "Failed to open: " << path << ".tmp" << std::endl;
	}
}

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

	if (argc < 2 || g_mode != op_mode::make_cache) {
		std::cerr << "VNSleuth v0.1" << std::endl;
	}

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <script_directory> <llama.cpp exec> [<optional llama.cpp args>...]" << std::endl;
		std::cerr << "TTY+no exec = print furigana, some information, and exit." << std::endl;
		std::cerr << "TTY+llama = generate translation cache." << std::endl;
		std::cerr << "pipe+no exec = print cached translation in real time (must be created)." << std::endl;
		std::cerr << "pipe+llama = translate in real time (not recommended)." << std::endl;
		std::cerr << "pipe+- = only send preprocessed lines to stdout." << std::endl;
		std::cerr << "Nothing to do." << std::endl;
		return 1;
	}

	// Create configuration directory
	std::string cfg_dir;
	if (const char* conf = ::getenv("XDG_CONFIG_HOME"))
		cfg_dir = std::string(conf) + "/vnsleuth/";
	else if (const char* home = ::getenv("HOME"))
		cfg_dir = std::string(home) + "/.config/vnsleuth/";
	else
		cfg_dir = "./config/";
	fs::create_directories(cfg_dir);

	// Create pipes for communication with the child process
	int opipe_fd[2]{}; // [0] Reading from llama.cpp stdout
	int ipipe_fd[2]{}; // [1] Writing to llama.cpp stdin
	int epipe_fd[2]{}; // [0] Reading from llama.cpp stderr
	if (g_mode == op_mode::rt_llama || g_mode == op_mode::make_cache) {
		if (pipe(opipe_fd) != 0 || pipe(ipipe_fd) != 0 || pipe2(epipe_fd, O_NONBLOCK) != 0) {
			perror("Failed to create pipes");
			return 1;
		}
	}

	const auto g_now = std::chrono::steady_clock::now();

	// Parse scripts in a given directory
	std::string line, llama_out, cache_path, prompt_path, names_path;
	std::size_t dummy_lines = 1;
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
	}

	if (!names_path.empty()) {
		// Load (pre-translated) names
		std::ifstream names(names_path);
		if (names.is_open()) {
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
	}

	// Sort file list alphabetically
	std::sort(file_list.begin(), file_list.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

	for (std::size_t i = 0; i < file_list.size(); i++) {
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

			std::ifstream cache(cache_path);
			if (parse(data, cache)) {
				// Add global delimiter
				dummy_lines++;
				g_text.emplace_back();
				g_cache.emplace_back();

				if (g_mode == op_mode::make_cache && !is_incomplete) {
					// Spawn process for each non-empty file
					auto& arg = file_list[i].first;
					argv[1] = arg.data();
					std::cerr << "Translating: " << arg << " -> " << cache_path << std::endl;
					if (cache.is_open()) {
						std::cerr << "File exists, skipping." << std::endl;
						continue;
					}
					pid_t pid = fork();
					if (pid == 0) {
						execvp(argv[0], argv);
						perror("execvp failed for vnsleuth:");
						return 1;
					}
					int wstatus = 0;
					waitpid(pid, &wstatus, 0);
					if (int res = WEXITSTATUS(wstatus)) {
						std::cerr << "Child process terminated abnormally: " << res << std::endl;
						return res;
					}
				}

				if (g_mode == op_mode::print_info || g_mode == op_mode::rt_cached) {
					if (g_mode == op_mode::print_info) {
						std::cerr << "Found file: " << file_list[i].first << std::endl;
						std::cerr << "Cache file: " << cache_path;
					}
					if (fs::is_regular_file(cache_path)) {
						if (g_mode == op_mode::print_info)
							std::cerr << " (exists)" << std::endl;
						if (!cache.is_open()) {
							std::cerr << "Error: Could not open translation cache: " << cache_path << std::endl;
							return 1;
						}
						if (g_cache.size() != g_text.size()) {
							std::cerr << "Error: Translation cache terminated abruptly: " << cache_path << std::endl;
							return 1;
						}
						std::getline(cache, line);
						if (!cache.eof()) {
							std::cerr << "Error: Translation cache exceeds expected size: " << cache_path << std::endl;
							return 1;
						}
					} else {
						if (g_mode == op_mode::print_info)
							std::cerr << " (not found)" << std::endl;
						is_incomplete = true;
					}
				}
			}
		} else if (file_list.size() == 1) {
			std::cerr << "Error: Could not open file: " << file_list[i].first << std::endl;
			return 1;
		}
	}

	if (g_mode != op_mode::make_cache || !is_incomplete) {
		std::cerr << "Loaded files: " << dummy_lines - 1 << std::endl;
		std::cerr << "Loaded lines: " << g_text.size() - dummy_lines << std::endl;
		std::cerr << "Loaded names: " << g_speakers.size() - 1 << std::endl;
	}

	if (g_mode == op_mode::make_cache && !is_incomplete) {
		const auto elaps = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - g_now);
		std::cerr << "Cache generation completed. Elapsed: " << elaps.count() << "s" << std::endl;
		return 0;
	}

	if (is_incomplete) {
		if (g_mode == op_mode::print_info || g_mode == op_mode::rt_cached)
			std::cerr << "Some cache files were not found." << std::endl;
	}

	if (g_mode == op_mode::print_info || g_text.size() <= 1) {
		// Print known furigana
		for (auto&& [type_as, read_as] : g_furigana) {
			std::cerr << type_as << "=" << read_as << std::endl;
		}

		// Update names
		if (g_text.size() > 1 && g_speakers.size() > 1) {
			update_names(names_path);
		}

		std::cerr << "Nothing to do." << std::endl;
		return 0;
	}

	// Generate args for llama.cpp
	std::vector<const char*> llama_args;
	for (int i = 2; i < argc; i++)
		llama_args.push_back(argv[i]);
	llama_args.push_back("--simple-io");
	llama_args.push_back("-f");
	llama_args.push_back(prompt_path.c_str()); // Specify prompt
	llama_args.push_back("--keep");
	llama_args.push_back("-1"); // Always keep prompt in the context
	llama_args.push_back("-n");
	llama_args.push_back("128"); // Generate 128 tokens
	llama_args.push_back("--temp");
	llama_args.push_back("0.2"); // Set low temperature
	llama_args.push_back("--top-p");
	llama_args.push_back("0.3");
	llama_args.push_back("--repeat-last-n");
	llama_args.push_back("3");
	llama_args.push_back("--repeat-penalty");
	llama_args.push_back("1.1"); // Penalize few last tokens
	llama_args.push_back("--ignore-eos");
	llama_args.push_back("--interactive-first");
	llama_args.push_back("-r");
	llama_args.push_back("\n"); // Stop on newline
	llama_args.push_back("--in-prefix");
	llama_args.push_back(iprefix.c_str());
	//llama_args.push_back("--grammar");
	//llama_args.push_back("root ::= [- a-zA-Z0-9~=!?.,:;'\"%&()\\n]+");
	llama_args.push_back("--logit-restrict");
	llama_args.push_back(" 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~-=!?.,:;'\"%&()\n");
	llama_args.push_back(nullptr);

	// Fork a child process if requested
	pid_t pid = argc >= 3 && argv[2] != "-"sv ? fork() : 1;
	if (pid == 0) {
		// Inside the child process
		while ((dup2(ipipe_fd[0], STDIN_FILENO) == -1) && (errno == EINTR)) {}
		while ((dup2(opipe_fd[1], STDOUT_FILENO) == -1) && (errno == EINTR)) {}
		while ((dup2(epipe_fd[1], STDERR_FILENO) == -1) && (errno == EINTR)) {}
		close(ipipe_fd[0]);
		close(ipipe_fd[1]);
		close(opipe_fd[0]);
		close(opipe_fd[1]);
		close(epipe_fd[0]);
		close(epipe_fd[1]);

		// Execute llama.cpp (main)
		execvp(llama_args[0], const_cast<char**>(llama_args.data()));
		perror("execvp failed for llama.cpp");
		return 1;
	} else if (pid > 1) {
		close(ipipe_fd[0]);
		close(opipe_fd[1]);
		close(epipe_fd[1]);
	}

	// Read from the pipe until JP: (iprefix) is encountered
	auto read_translated = [&]() -> bool {
		char buf{};
		while (true) {
			auto r = read(opipe_fd[0], &buf, 1);
			if (r <= 0) {
				if (r < 0) {
					perror("Reading from pipe failed");
				}
				std::string err;
				err.resize(10000);
				err.resize(std::max<ssize_t>(0, read(epipe_fd[0], err.data(), err.size())));
				std::cerr << err << std::flush;
				return false;
			}
			llama_out += buf;
			static const auto real_iprefix = "\n" + iprefix;
			if (llama_out.ends_with(real_iprefix))
				return true;
		}
	};

	if (pid > 1) {
		// Initialize translation
		llama_out += "SRC:";
		llama_out += argv[1];
		llama_out += "\n";
		if (!read_translated()) {
			std::cerr << "Translation prompt not received." << std::endl;
			return 1;
		}
	}

	if (g_mode == op_mode::make_cache) {
		// Send all lines in a script sequentially for translation
		const auto _now = std::chrono::steady_clock::now();
		const auto nlines = g_text.size() - 2;
		std::ofstream cache(cache_path + ".tmp", std::ios::trunc);
		if (!cache.is_open()) {
			std::cerr << "Failed to open file " << cache_path << ".tmp" << std::endl;
			return 1;
		}
		std::cerr << "Translated lines: 0/" << nlines << std::flush;
		for (std::size_t i = 1; i < nlines + 1; i++) {
			const auto spker = print_line(i, ipipe_fd[1], &llama_out, i == 1 ? example : "");
			cache << llama_out;
			llama_out.clear();
			if (!read_translated()) {
				return 1;
			}
			if (spker.size()) {
				// Try to parse translated name
				std::string_view speaker = llama_out;
				if (auto pos = speaker.find_first_not_of(" ") + 1)
					speaker.remove_prefix(pos - 1);
				speaker = speaker.substr(0, speaker.find_first_of(':') + 1);
				if (speaker.empty() || speaker.find_first_of('"') + 1) {
					std::cerr << "Failed to parse speaker translation for: " << spker << std::endl;
					g_speakers[spker] = ":";
				} else {
					g_speakers[spker] = speaker;
				}
			}
			REPLACE(llama_out, "\r", "");
			REPLACE(llama_out, "\n\n", "\n"); // Clear empty lines (although should be unnecessary)
			cache << llama_out;
			llama_out.clear();
			const auto stamp = std::chrono::steady_clock::now();
			const auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stamp - _now);
			std::cerr << "\rTranslated lines: " << i << "/" << nlines << "... (" << elaps.count() / i / 1000. << "s/line)    " << std::flush;
		}
		cache << std::flush;
		cache.close();
		update_names(names_path);
		fs::rename(cache_path + ".tmp", cache_path);
		const auto elaps = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - g_now);
		std::cerr << std::endl << "Elapsed: " << elaps.count() << "s" << std::endl;
		kill(pid, SIGTERM);
		waitpid(pid, nullptr, 0);
		return 0;
	} else {
		std::cout << llama_out << std::flush;
		llama_out.clear();
	}

	std::cerr << "Waiting for input..." << std::endl;

	// Hint for text search (predicted entry, 0 is "impossible" default which points to pad entry)
	size_t next_id = 0, prev_id = 0;

	// Exact match mode:
	// When line is found and is expected, it's printed out (back_buf remains empty).
	// When line is found and is unique, the back buffer is flushed as well.
	// When line is found but not unique, it's added to the back buffer.
	// Previous non-unique lines are printed if they exactly precede unique line, otherwise wiped.
	std::vector<std::string> back_buf;
	while (std::getline(std::cin, line)) {
		const auto it = g_loc.find(line);
		if (line.empty() || it == g_loc.end()) {
			// Line not found (garbage?)
			//std::cerr << "Ignored line: " << line << std::endl;
			continue;
		}
		size_t last_id = next_id;
		if (g_text[next_id].second != line) {
			// Handle unexpected line:
			if (it->second == 0) {
				// Remember ambiguous line
				back_buf.emplace_back(std::move(line));
				next_id = 0;
				continue;
			}

			if (it->second == prev_id) {
				// Avoid repeating previous line
				back_buf.clear();
				next_id = prev_id + 1;
				continue;
			}

			next_id = it->second;
			last_id = it->second;

			// Restore back log:
			for (auto itr = back_buf.rbegin(); itr != back_buf.rend(); itr++) {
				bool found = false;
				// Compensate for repetition suppression
				while (g_text[next_id - 1].second == *itr) {
					found = true;
					next_id--;
				}
				if (!found)
					break;
			}

			back_buf.clear();
		}

		// Print line(s)
		while (next_id <= last_id) {
			static const bool tty_out = !!isatty(STDOUT_FILENO);
			if (g_mode == op_mode::print_only) {
				print_line(next_id);
			} else if (g_mode == op_mode::rt_llama) {
				print_line(next_id, ipipe_fd[1], &llama_out, example);
				example = ""; // Only inject example before first line
				std::cout << llama_out << std::flush;
				llama_out.clear();
				if (!read_translated())
					return 1;
				std::cout << llama_out << std::flush;
				llama_out.clear();
			} else if (g_mode == op_mode::rt_cached) {
				if (g_cache[next_id].empty()) {
					std::cerr << "Error: Translation cache is incomplete" << std::endl;
				} else {
					auto& str = g_cache[next_id];
					if (tty_out) {
						llama_out = str;
						if (!llama_out.starts_with(iprefix)) {
							std::cerr << iprefix << " not found in translation cache." << std::endl;
							return 1;
						}
						llama_out.replace(0, iprefix.size(), "\033[0;33m");
						static const auto real_isuffix = "\n" + isuffix + (isuffix.ends_with(" ") ? "" : " ");
						const auto suff_pos = llama_out.find(real_isuffix);
						if (suff_pos + 1 == 0) {
							std::cerr << isuffix << " not found in translation cache." << std::endl;
							return 1;
						}
						llama_out.replace(suff_pos + 1, real_isuffix.size() - 1, "\033[0;1m");
						std::cout << llama_out << "\033[0m" << std::flush;
					} else {
						std::cout << str << std::flush;
					}
				}
			}

			prev_id = next_id++;

			// Auto-continuation of repeated lines (compensate for repetition suppression)
			if (next_id > last_id && g_text[next_id].second == g_text[prev_id].second) {
				last_id++;
				continue;
			}

			// List choices that are supposed to appear on the screen
			if (next_id > last_id && g_text[next_id].first.starts_with("選択肢#")) {
				last_id++;
				continue;
			}
		}
	}
	return 0;
}
