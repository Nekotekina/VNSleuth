#include "main.hpp"
#include "tools/iconv.hpp"
#include <bit>
#include <functional>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

static /*thread_local*/ std::function<std::ostream&()> err = [] { return std::ref(std::cerr); };

// Read little-endian number from string
template <typename T, typename Off>
bool read_le(T& dst, std::string_view data, Off&& pos)
{
	static_assert(std::endian::native == std::endian::little, "Big Endian platform support not implemented");
	if (pos + sizeof(T) > data.size())
		return false;
	std::memcpy(&dst, data.data() + pos, sizeof(T));
	if constexpr (!std::is_const_v<std::remove_reference_t<Off>>)
		pos += sizeof(T); // Optionally increment position
	return true;
}

// Read null-terminated Shift-JIS string from string
bool read_sjis(std::string& dst, std::string_view data, std::size_t pos)
{
	static const iconvpp::converter conv("UTF-8", "SJIS-WIN", true, 1024);
	if (pos >= data.size())
		return false;
	dst.clear();
	conv.convert(data.data() + pos, dst);
	return true;
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

// Return number of lines parsed
uint parse(std::string_view data)
{
	uint result = 0;

	// Detect script format then parse appropriately
	const bool is_text = is_text_bytes(data);

	if (!is_text && data.starts_with("BurikoCompiledScriptVer1.00\0"sv)) {
		std::uint32_t add_off{}, off{}, op{};
		if (!read_le(add_off, data, 0x1c))
			return false;
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
		while (read_le(op, data, off)) {
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
				std::uint32_t addr{};
				if (!read_le(addr, data, off))
					break;
				std::string buf;
				if (read_sjis(buf, data, addr + add_off)) {
					stack.emplace_back(std::move(buf));
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
				result++;
			}

			stack.clear();
			continue;
		}

		return result;
	}

	return 0;
}
