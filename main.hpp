#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std::literals;

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

using uint = unsigned;

// Operation mode
inline enum class op_mode {
	print_only = 0, // Legacy mode (passthrough)
	print_info,		// Check only
	rt_cached,
	rt_llama,
} g_mode{};

struct line_info {
	std::string name;		  // Character name (speaker), may be empty
	std::string text;		  // Original text
	std::string_view sq_text; // Processed text (squeezed, owned permanently by g_strings)
	std::string tr_text;	  // Translated text (two-line format)
	uint seed = 0;			  // Increases with each rewrite
};

struct segment_info {
	std::vector<line_info> lines;
	std::string src_name;
	std::string cache_name;
	std::vector<std::string> prev_segs; // For history extraction
	std::string tr_tail;
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
	std::unordered_map<std::string, uint> segs_by_name;

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

	// Check id is valid and last in the segment
	bool is_last(line_id id) const noexcept
	{
		if (id.first >= segs.size())
			return false;
		if (id.second + 1 && id.second == segs[id.first].lines.size() - 1)
			return true;
		return false;
	}
} g_lines;

// String database for search (squeezed string -> line_id)
inline std::unordered_map<std::string, line_id> g_strings;

// Auxiliary string search helper, only contains first lines in each segment
inline std::unordered_map<std::string, line_id> g_start_strings;

// Furigana database (word ; reading)
inline std::set<std::pair<std::string, std::string>> g_furigana;

// Speaker database (name -> translation)
inline std::map<std::string, std::string, std::less<>> g_speakers{{"？？？:", "???:"}};

// Default replace rules to reduce token count and also fix potential issues with borked translations
inline const std::vector<std::pair<std::string, std::string>> g_default_replaces{
	// clang-format off
	{"\t", "　"},
	{"〜", "～"}, // Use one more typical for SJIS-WIN
	{"……", "…"},
	{"──", "─"},
	{"――", "―"},
	{"ーーー", "ーー"}, // Reduce repetitions
	{"ーー", "～"}, // Replace "too long" sound with ～
	{"～～", "～"},
	{"「…", "「"}, // Sentences starting with … might get translated as empty "..."
	{"（…", "（"},
	{"『…", "『"},
	{"？？", "？"},
	{"ぁぁ", "ぁ"},
	{"ぃぃ", "ぃ"},
	{"ぅぅ", "ぅ"},
	{"ぇぇ", "ぇ"},
	{"ぉぉ", "ぉ"},
	{"ゃゃ", "ゃ"},
	{"ゅゅ", "ゅ"},
	{"ょょ", "ょ"},
	{"っっ", "っ"},
	{"ァァ", "ァ"},
	{"ィィ", "ィ"},
	{"ゥゥ", "ゥ"},
	{"ェェ", "ェ"},
	{"ォォ", "ォ"},
	{"ャャ", "ャ"},
	{"ュュ", "ュ"},
	{"ョョ", "ョ"},
	{"ッッ", "ッ"},
	// clang-format on
};

// Additional replace rules from __vnsleuth_replace.txt
inline std::vector<std::pair<std::string, std::string>> g_replaces = g_default_replaces;

// Apply replace rules
inline std::string apply_replaces(const auto& src, bool repeatable_only = false, std::size_t start = 0)
{
	std::string out;
	out = src;
	for (auto& [from, to] : g_replaces) {
		std::size_t start_pos = start;
		std::size_t start_inc = to.rfind(from);
		if (repeatable_only && start_inc + 1)
			continue;
		while (auto pos = out.find(from, start_pos) + 1) {
			out.replace(pos - 1, from.size(), to);
			if (start_inc + 1) {
				// Perform only single replacement if `to` contains `from`
				start_pos = pos - 1 + start_inc + from.size();
			} else {
				start_pos = pos - 1;
			}
		}
	}
	return out;
}

// Global history
inline std::vector<line_id> g_history;

// Global mutex
inline std::shared_mutex g_mutex;

// Remove all repeating characters in line
std::string squeeze_line(const std::string& line);

// translate() control
enum class tr_cmd {
	translate, // Translate line in foreground
	kick,	   // Start background worker
	sync,	   // Terminate background worker
	eject,	   // Eject id.second last lines
	reload,	   // Reload starting from id
};

// Obvious.
bool translate(struct gpt_params& params, line_id id, tr_cmd cmd = tr_cmd::translate);

// Terminal colors
inline struct escape_sequences {
	const char reset[16] = "\033[0m";
	const char orig[16] = "\033[0;33m"; // yellow
	const char tran[16] = "\033[0;1m";	// bold

	void disable()
	{
		// Clear all strings
		static_assert(std::is_trivially_copyable_v<escape_sequences>);
		std::memset(this, 0, sizeof(*this));
	}
} g_esc{};
