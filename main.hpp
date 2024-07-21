#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std::literals;

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

using uint = unsigned;

// Operation mode
inline enum class op_mode {
	print_only = 0, // Legacy mode (passthrough)
	rt_cached,
	rt_llama,
	print_info,
	make_cache, // Offline cache generation
} g_mode{};

// Text line information (lines are owned by g_strings)
struct line_info {
	std::string name;		  // Character name (speaker), may be empty
	std::string_view text;	  // Line text
	std::string tr_text;	  // Translated text (two-line format)
	uint seed = 0;			  // Increases with each rewrite
};

struct segment_info {
	std::vector<line_info> lines;
	std::string src_name;
	std::string cache_name;
};

// Line location (segment and index)
using line_id = std::pair<uint, uint>;

// Bad line id for ambiguous lines
static constexpr line_id c_bad_id{UINT32_MAX, UINT32_MAX};

#define REPLACE(s, x, y)                                                                                                                             \
	while (auto pos = s.find(x##sv, 0) + 1)                                                                                                          \
		s.replace(pos - 1, x##sv.size(), y##sv);

// 2D database of lines
inline struct loaded_lines {
	std::vector<segment_info> segs;

	line_info& operator[](line_id id)
	{
		// Throw if out of range
		return segs.at(id.first).lines.at(id.second);
	}

	const line_info& operator[](line_id id) const
	{
		// Throw if out of range
		return segs.at(id.first).lines.at(id.second);
	}

	line_id last() const noexcept
	{
		if (segs.empty())
			return c_bad_id;
		if (segs.back().lines.empty())
			return c_bad_id;
		line_id r{};
		r.first = segs.size() - 1;
		r.second = segs.back().lines.size() - 1;
		return r;
	}

	// Get next line id
	line_id next(line_id id) const noexcept
	{
		if (!is_valid(id))
			return c_bad_id;
		id.second++;
		if (!is_valid(id))
			return c_bad_id;
		return id;
	}

	// Get next line id
	bool advance(line_id& id) const noexcept
	{
		if (!is_valid(id)) {
			id = c_bad_id;
			return false;
		}
		id.second++;
		if (!is_valid(id)) {
			id = c_bad_id;
			return false;
		}
		return true;
	}

	// Count all lines in segments
	std::size_t count_lines() const noexcept
	{
		std::size_t r = 0;
		for (auto& s : segs)
			r += s.lines.size();
		return r;
	}

	// Check id for validity
	bool is_valid(line_id id) const noexcept
	{
		if (id.first >= segs.size())
			return false;
		if (id.second >= segs[id.first].lines.size())
			return false;
		return true;
	}
} g_lines;

// String database (string -> line_id) which owns strings.
inline std::unordered_map<std::string, line_id> g_strings;

// Furigana database (word ; reading)
inline std::set<std::pair<std::string, std::string>> g_furigana;

// Speaker database (name -> translation)
inline std::map<std::string, std::string, std::less<>> g_speakers{{"？？？:", "???:"}};

// Parse script into global variables; return number of lines parsed
uint parse(std::string_view data);
