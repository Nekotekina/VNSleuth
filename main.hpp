#pragma once

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std::literals;

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

// Operation mode
inline enum class op_mode {
	print_only = 0, // Legacy mode (passthrough)
	rt_cached,
	rt_llama,
	print_info,
	make_cache,
} g_mode{};

#define REPLACE(s, x, y)                                                                                                                             \
	while (auto pos = s.find(x##sv, 0) + 1)                                                                                                          \
		s.replace(pos - 1, x##sv.size(), y##sv);

// Text pairs (character name ; text), empty text is chapter padding.
inline std::vector<std::pair<std::string_view, std::string_view>> g_text{1};

// String database (string -> text position) which also owns strings.
inline std::unordered_map<std::string, size_t> g_loc;

// Translation cache (corresponds to g_text)
inline std::vector<std::string> g_cache{1};

// Furigana database (word -> reading)
inline std::set<std::pair<std::string, std::string>> g_furigana;

// Speaker database (name -> translation)
inline std::map<std::string, std::string, std::less<>> g_speakers{{"？？？:", "???:"}};

// Parse script into global variables; return number of lines parsed
std::size_t parse(const std::string& data, std::istream& cache);
