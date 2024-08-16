#include "parser.hpp"
#include "main.hpp"
#include "tools/iconv.hpp"
#include "tools/tiny_sha1.hpp"
#include <bit>
#include <functional>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

static /*thread_local*/ std::function<std::ostream&()> err = [] { return std::ref(std::cerr); };

// Convert from Windows-specific Shift-JIS encoding to UTF-8
std::string from_sjis(std::string_view src)
{
	static const iconvpp::converter conv("UTF-8", "SJIS-WIN", true, 1024);
	std::string dst;
	conv.convert(src.data(), dst);
	return dst;
}

bool is_text_bytes(std::string_view data)
{
	for (unsigned char c : data) {
		// Find unexpected control characters
		if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
			return false;
		}
		if (c == 0x7f) {
			return false;
		}
	}

	return true;
}

void add_segment(const std::string& name, std::string_view data)
{
	if (!g_lines.segs.empty() && g_lines.segs.back().lines.empty()) {
		g_lines.segs_by_name.erase(g_lines.segs.back().src_name);
		g_lines.segs.pop_back();
	}

	auto& new_seg = g_lines.segs.emplace_back();
	new_seg.src_name = name;
	{
		// Generate SHA1 hash for cache file name
		// If some game update changes the script, it will result in creating a different cache file
		char buf[42]{};
		sha1::SHA1 s;
		s.processBytes(data.data(), data.size());
		std::uint32_t digest[5];
		s.getDigest(digest);
		std::snprintf(buf, 41, "%08x%08x%08x%08x%08x", digest[0], digest[1], digest[2], digest[3], digest[4]);
		new_seg.cache_name += "__vnsleuth_";
		new_seg.cache_name += buf;
		new_seg.cache_name += ".txt";
	}

	// TODO: handle duplicates correctly
	if (g_lines.segs_by_name.count(name))
		throw std::runtime_error("Segment already exists: " + name);
	g_lines.segs_by_name[name] = g_lines.segs.size() - 1;
}

void add_line(int choice, std::string name, std::string text)
{
	if (!is_text_bytes(name))
		err() << "Bad line (name): " << name << std::endl;
	if (!is_text_bytes(text))
		err() << "Bad line (text): " << text << std::endl;

	// Apply some filtering: remove \r and replace \n with \t
	REPLACE(text, "\r", "");
	while (auto pos = text.find_first_of("\n") + 1)
		text[pos - 1] = '\t';
	// Replace special ":" character with full-width variant
	REPLACE(text, ":", "：");

	if (!name.empty()) {
		// Add (special) character ":" after name
		REPLACE(name, "\r", "");
		REPLACE(name, "\n", " ");
		REPLACE(name, ":", "：");
		REPLACE(name, "#", "＃");
		REPLACE(name, "　", " "); // Temporarily use ASCII spaces; trimming
		while (name.ends_with(" "))
			name.erase(name.end() - 1);
		while (name.starts_with(" "))
			name.erase(0, 1);
		REPLACE(name, " ", "　"); // Change all spaces to full-width
		if (name.empty()) {
			// Use placeholder for empty names
			name = "？？？";
		}

		name += ":";
	}

	if (choice && name.empty()) {
		// Add choice "name" in special format
		name = "選択肢#" + std::to_string(choice) + ":";
	}

	line_id id{};
	id.first = g_lines.segs.size() - 1;
	id.second = g_lines.segs[id.first].lines.size();

	auto [text_it, ok] = g_strings.emplace(squeeze_line(text), id);
	if (!ok)
		text_it->second = c_bad_id; // Mark if not unique

	line_info& line = g_lines.segs[id.first].lines.emplace_back();
	line.name = std::move(name);
	line.text = std::move(text);
	line.sq_text = std::as_const(text_it->first);

	// Remember all encountered characters in a bitmap (doesn't include names)
	for (char32_t c : text_it->first)
		g_chars.set(c);

	// Check max length of squeezed string
	if (text_it->first.size() > 255)
		throw std::runtime_error("Line too long: " + line.text);

	if (id.second == 0) {
		auto [it, ok2] = g_start_strings.emplace(text_it->first, id);
		if (!ok2)
			it->second = c_bad_id;
	}

	if (!line.name.empty() && !choice)
		g_speakers.emplace(line.name, std::string());
}

// Add furigana
void add_ruby(std::string type_as, std::string read_as)
{
	// Don't register accent dots as furigana
	static constexpr std::string_view rep_dot = "・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・"
												"・・・・・・・・・・・・・・・・・・・・";
	if (rep_dot.starts_with(read_as))
		return;
	g_furigana.emplace(std::move(type_as), std::move(read_as));
}

// Parse and register furigana (Buriko/Ethornell engine)
std::string parse_ruby_eth(const std::string& text)
{
	static const std::regex rrgx("<R(.*?)>(.*?)</R>");
	auto words_begin = std::sregex_iterator(text.begin(), text.end(), rrgx);
	auto words_end = std::sregex_iterator();
	std::size_t shift = 0;
	std::string result = text;

	for (std::sregex_iterator i = words_begin; i != words_end; i++) {
		auto read_as = i->str(1);
		auto type_as = i->str(2);

		result.replace(i->position() - shift, i->length(), type_as);
		shift += i->length() - type_as.length();
		add_ruby(std::move(type_as), std::move(read_as));
	}

	if (result.find_first_of("<") + 1)
		err() << "Possibly unknown tag found" << std::endl;

	// Remove newlines completely
	while (auto pos = result.find_first_of("\n\r") + 1)
		result.erase(pos - 1, 1);
	return result;
}

// To avoid parsing all archives, limit the set of possible candidates
static const std::unordered_set<std::string_view> script_locations{
	{"data01000.arc"},
};

void script_parser::read_segments(const std::string& name)
{
	// Detect script format then parse appropriately
	const bool is_text = is_text_bytes(data);

	std::multimap<std::string, parser_func_t, natural_order_less> sorted_archive;
	if (!is_text && script_locations.count(name)) {
		auto arc = read_archive();
		sorted_archive = decltype(sorted_archive)(std::make_move_iterator(arc.begin()), std::make_move_iterator(arc.end()));
	}

	if (!is_text && !sorted_archive.empty()) {
		for (auto& [n, f] : sorted_archive) {
			script_parser parser(f(true, nullptr).data);
			//std::ofstream("__vnsleuth_dump_" + n, std::ios::trunc | std::ios::binary).write(parser.data.data(), parser.data.size());
			parser.read_segments(n);
		}
		return;
	}

	if (!is_text && data.starts_with("BurikoCompiledScriptVer1.00\0"sv)) {
		std::uint32_t add_off{}, off{}, op{};
		if (!read_le(add_off, 0x1c) || data.size() < add_off)
			return;
		add_segment(name, data);
		off = (add_off += 0x1c); // Offset added to PUSH opcode string location, and first OP offset
		std::vector<std::string> stack;
		err = [&] {
			std::cerr << "Format: Buriko/ETH" << std::endl;
			std::cerr << "Offset: " << off << std::endl;
			std::cerr << "Opcode: " << op << std::endl;
			std::cerr << "Stack: " << stack.size() << std::endl;
			for (auto& s : stack)
				std::cerr << s << std::endl;
			std::cerr << "Error: ";
			return std::ref(std::cerr);
		};
		while (read_le(op, off)) {
			switch (op) {
			case 0:
			case 1:
			case 2:
			case 8:
			case 9:
			case 0xa:
			case 0x17:
			case 0x19:
			case 0x3f:
			case 0x7e:
				off += 4; // Skip args
				continue;
			case 0x7f:
				off += 8; // Skip args
				continue;
			case 0x7b:
				off += 12; // Skip args
				continue;
			default:
				break;
			}
			if (op == 0xf4) {
				// EXIT?
				break;
			}
			if (op == 3) {
				// PUSH
				auto [addr, ok] = read_le<4u>(off);
				if (!ok)
					break;
				auto [str, ok2] = read_le<0>(addr + add_off);
				if (ok2) {
					stack.emplace_back(from_sjis(str));
				} else {
					stack.clear(); // TODO
				}
				continue;
			}
			if (op == 28 && !stack.empty()) {
				// CALL
				if (stack.back() == "_Selection") {
					if (stack.size() < 3)
						err() << "_Selection: too little stack" << std::endl;

					// Add choices
					for (unsigned i = 1; i < stack.size(); i++) {
						add_line(i, "", parse_ruby_eth(stack[i - 1]));
					}
				}
			}
			if (op == 0x140 && !stack.empty()) {
				// PRINT
				std::string name, text, textfix;
				text = stack.back();
				if (stack.size() > 1) {
					name = stack[0];
				}

				if (stack.size() > 2)
					err() << "PRINT: too big stack (parse error)" << std::endl;

				REPLACE(text, "\01", ""); // Found in Sekachu
				textfix = parse_ruby_eth(text);

				// Add normal line
				add_line(0, std::move(name), std::move(textfix));
			}

			stack.clear();
			continue;
		}
	}
}
