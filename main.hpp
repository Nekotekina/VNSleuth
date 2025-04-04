#pragma once

#include <atomic>
#include <bitset>
#include <chrono>
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
using ui64 = std::uint64_t;

struct natural_order_less {
	constexpr bool operator()(std::string_view a, std::string_view b) const
	{
		constexpr auto decimal = "0123456789"sv;
		while (!a.empty()) {
			auto ac = a.substr(0, a.find_first_of(decimal));
			auto bc = b.substr(0, b.find_first_of(decimal));
			a.remove_prefix(ac.size());
			b.remove_prefix(bc.size());
			if (auto c = ac <=> bc; c != 0) // Compare non-numeric fragments
				return c < 0;
			auto an = a.substr(0, a.find_first_not_of(decimal));
			auto bn = b.substr(0, b.find_first_not_of(decimal));
			a.remove_prefix(an.size());
			b.remove_prefix(bn.size());
			auto ans = an.size();
			auto bns = bn.size();
			while (an.starts_with("0"))
				an.remove_prefix(1);
			while (bn.starts_with("0"))
				bn.remove_prefix(1);
			if (auto c = an.size() <=> bn.size(); c != 0) // Compare numbers by length (01 still equals 1)
				return c < 0;
			if (auto c = an <=> bn; c != 0) // Compare numbers of equal length as strings
				return c < 0;
			if (auto c = ans <=> bns; c != 0) // Finally, compare equal numbers by string length (1 < 01)
				return c < 0;
		}
		return !b.empty();
	}
};

template <typename Char, typename Tr, typename All, typename T>
std::basic_string<Char>& operator-=(std::basic_string<Char, Tr, All>& str, const T& tail)
	requires(std::is_convertible_v<T, std::basic_string_view<Char, Tr>>)
{
	std::basic_string_view<Char, Tr> t = tail;
	if (str.ends_with(t))
		str.resize(str.size() - t.size());
	return str;
}

template <typename Src, typename T, typename Char = std::decay_t<decltype(std::declval<Src>()[0])>>
auto operator-(const Src& sv, const T& tail)
	requires(std::is_trivial_v<Char> && std::is_standard_layout_v<Char> && std::is_constructible_v<std::basic_string_view<Char>, const Src&> &&
			 std::is_constructible_v<std::basic_string_view<Char>, const T&>)
{
	std::basic_string_view<Char> s(sv);
	std::basic_string_view<Char> t(tail);
	if (s.ends_with(t))
		s.remove_suffix(t.size());
	return s;
}

template <typename Char, typename Tr, typename T>
std::basic_string_view<Char>& operator-=(std::basic_string_view<Char, Tr>& sv, const T& tail)
	requires(std::is_convertible_v<T, std::basic_string_view<Char, Tr>>)
{
	std::basic_string_view<Char, Tr> t = tail;
	if (sv.ends_with(t))
		sv.remove_suffix(t.size());
	return sv;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& vec, const std::vector<T>& vec_add)
{
	for (auto& e : vec_add)
		vec.emplace_back(e);
	return vec;
}

// Operation mode
inline enum class op_mode {
	print_only = 0, // Legacy mode (passthrough)
	print_info,		// Check only
	rt_cached,
	rt_llama,
} g_mode{};

// Line location (segment and index)
using line_id = std::pair<uint, uint>;

struct line_info {
	std::string name;		  // Character name (speaker), may be empty
	std::string text;		  // Original text
	std::u16string_view sq_text; // Processed text (squeezed, owned permanently by g_strings)
	std::string tr_text;		 // Translated text (includes original line and annotations)
	std::string pre_ann;
	std::string post_ann;
	std::vector<std::vector<int>> tr_tts; // tr_text tokens (only translated part)
	uint tokens;
	uint segment;
	std::vector<float> embd;
	double embd_sqrsum;

	line_id get_id() const;
};

struct segment_info {
	std::vector<line_info> lines;
	std::string src_name;
	std::string cache_name;
	std::vector<std::string> prev_segs; // For history extraction
	std::string tr_tail;
};

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

	struct iterator final : private line_id {
		constexpr iterator() : line_id(c_bad_id) {}
		constexpr explicit iterator(line_id id) : line_id(id) {}

		constexpr auto operator<=>(const iterator&) const = default;

		iterator operator++(int) noexcept;
		iterator& operator++() noexcept;

		line_info& operator*() const;
		line_info* operator->() const { return &**this; }

		line_id id() const { return *this; }
	};

	iterator begin() const { return iterator{{0u, 0u}}; }
	iterator end() const noexcept { return {}; }
} g_lines;

inline loaded_lines::iterator loaded_lines::iterator::operator++(int) noexcept
{
	iterator r = *this;
	this->operator++();
	return r;
}

inline loaded_lines::iterator& loaded_lines::iterator::operator++() noexcept
{
	if (*this != c_bad_id) {
		if (g_lines.is_last(*this)) {
			this->first++;
			this->second = 0;
			if (this->first >= g_lines.segs.size()) {
				this->first = -1;
				this->second = -1;
			}
		} else {
			this->second++;
		}
	}
	return *this;
}

inline line_info& loaded_lines::iterator::operator*() const
{
	// May throw
	return g_lines[*this];
}

inline line_id line_info::get_id() const
{
	line_id r;
	r.first = segment;
	r.second = this - g_lines.segs[segment].lines.data();
	return r;
}

// String database for search (squeezed string -> line_id)
inline std::unordered_map<std::u16string, line_id> g_strings;

// Auxiliary string search helper, only contains first lines in each segment
inline std::unordered_map<std::u16string, line_id> g_start_strings;

// Map of all encountered characters (by char16_t max value)
// Characters outside of first plane will be truncated
inline std::bitset<0x10000> g_chars{};

// Furigana database (word ; reading)
inline std::set<std::pair<std::string, std::string>> g_furigana;

// Dictionary (name -> translation; annotation)
inline std::map<std::string, std::pair<std::string, std::string>, std::less<>> g_dict{{"？？？:", {"???:", ""}}};

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
inline std::string apply_replaces(const auto& src, bool repeatable_only, auto&& start)
{
	std::string out;
	out = src;
	std::size_t max_pos = 0;
	for (auto& [from, to] : g_replaces) {
		std::size_t start_pos = start;
		std::size_t start_inc = to.rfind(from);
		if (repeatable_only && start_inc + 1)
			continue;
		while (auto pos = out.find(from, start_pos) + 1) {
			out.replace(pos - 1, from.size(), to);
			max_pos = std::max(pos, max_pos);
			if (start_inc + 1) {
				// Perform only single replacement if `to` contains `from`
				start_pos = pos - 1 + start_inc + from.size();
			} else {
				start_pos = pos - 1;
			}
		}
	}
	if constexpr (!std::is_const_v<std::remove_reference_t<decltype(start)>>)
		start = std::max<std::size_t>(start, max_pos);
	return out;
}

// Global history
inline std::vector<line_id> g_history;

// Global mutex
inline std::mutex g_mutex;

// Remove all repeating characters in line
std::u16string squeeze_line(const std::string& line);

// translate() control
enum class tr_cmd {
	translate, // Translate line in foreground
	kick,	   // Start background worker
	sync,	   // Terminate background worker
	eject,	   // Eject id.second last lines
	reload,	   // Reload starting from id
	unload,	   // On exit
};

// Obvious.
bool translate(struct common_params& params, line_id id, tr_cmd cmd = tr_cmd::translate);

struct alignas(4096) vnsleuth_stats {
	std::atomic<ui64> start_time;
	std::atomic<ui64> last_time;

	std::atomic<ui64> rt_reading_ms; // Approximate user time spent reading-translating (in ms)
	std::atomic<ui64> rt_afk_ms[31]; // Big AFK timespans are added here depending on log2(duration)
	std::atomic<ui64> rt_rewrites;	 // Possibly indicates the amount of automated edits
	std::atomic<ui64> rt_reloads;	 // Possibly indicates the amount of manual edits

	std::atomic<ui64> raw_translates; // Raw translation count
	std::atomic<ui64> raw_accepts;	  // +1 when next line in history is already translated in background
	std::atomic<ui64> raw_discards;	  // Raw discard count
	std::atomic<ui64> raw_samples;	  // Raw sample count
	std::atomic<ui64> raw_decodes;	  // Raw decoded count
	std::atomic<ui64> sample_count;
	std::atomic<ui64> sample_time; // In µs
	std::atomic<ui64> batch_count;
	std::atomic<ui64> batch_time; // In µs
	std::atomic<ui64> eval_time;  // In µs
	std::atomic<ui64> lag_time;	  // In µs
	std::atomic<ui64> lag_count;  // For eval_lag
};

static_assert(std::is_trivially_copyable_v<vnsleuth_stats>);
static_assert(alignof(vnsleuth_stats) >= sizeof(vnsleuth_stats));

// Points to a memory-mapped file
inline vnsleuth_stats* g_stats{};

// Terminal colors
inline const struct escape_sequences {
	mutable char reset[16] = "\033[0m";
	mutable char orig[16] = "\033[0;33m"; // yellow
	mutable char tran[16] = "\033[0;1m";  // bold
	mutable char buf[16] = "\033[0;37m";

	void disable() const
	{
		// Clear all strings
		static_assert(std::is_trivially_copyable_v<escape_sequences>);
		std::memset(const_cast<escape_sequences*>(this), 0, sizeof(*this));
	}
} g_esc{};
