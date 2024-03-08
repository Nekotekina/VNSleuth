#include "iconv.hpp"
#include "main.hpp"
#include <bit>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

extern const std::size_t example_lines;

// Read little-endian number from string
template <typename T, typename Off>
bool read_le(T& dst, const std::string& data, Off&& pos)
{
	static_assert(std::endian::native == std::endian::little, "Big Endian platform support not implemented");
	if (pos + sizeof(T) > data.size())
		return false;
	std::memcpy(&dst, data.data() + pos, sizeof(T));
	if constexpr (!std::is_const_v<Off>)
		pos += sizeof(T); // Optionally increment position
	return true;
}

// Read null-terminated Shift-JIS string from string
bool read_sjis(std::string& dst, const std::string& data, size_t pos)
{
	static const iconvpp::converter conv("UTF-8", "SJIS-WIN", true, 1024);
	if (pos >= data.size())
		return false;
	dst.clear();
	conv.convert(data.c_str() + pos, dst);
	return true;
}

void add_line(std::string name, std::string text, std::istream& cache)
{
	// Apply some filtering: remove \r and replace \n with \t
	while (auto pos = text.find_first_of("\r") + 1)
		text.erase(pos - 1, 1);
	while (auto pos = text.find_first_of("\n") + 1)
		text[pos - 1] = '\t';

	if (!name.empty()) {
		// Add (special) character ":" after name
		REPLACE(name, ":", "：");
		name += ":";
	}

	auto [name_it, ok1] = g_loc.try_emplace(std::move(name), 0);
	auto [text_it, ok2] = g_loc.try_emplace(std::move(text), g_text.size());
	name_it->second = 0; // Names are never unique
	if (!ok2)
		text_it->second = 0; // Mark if not unique

	g_text.emplace_back(name_it->first, text_it->first);
	if (name_it->first.size())
		g_speakers.emplace(name_it->first, std::string());
	if (cache) {
		// Extract cached translation
		auto getline = [&](std::string& src, std::string& dst) -> bool {
			const bool started = cache.tellg() == 0;
			std::string temp;
			while (std::getline(cache, temp)) {
				// Skip prompt (or garbage?)
				if (temp.starts_with("JP:")) {
					break;
				}
			}
			if (started) {
				// Skip example
				for (std::size_t i = 0; i < example_lines; i++) {
					if (!std::getline(cache, temp))
						return false;
				}
			}

			if (!temp.starts_with("JP:")) {
				return false;
			}
			src = std::move(temp);
			if (!std::getline(cache, dst)) {
				return false;
			}

			return true;
		};

		// Store both JP and EN lines as a single string
		if (getline(name, text)) {
			name += '\n';
			name += text;
			name += '\n';
			g_cache.emplace_back(std::move(name));
		}
	} else {
		// Add empty translation line
		g_cache.emplace_back();
	}
}

// Return number of lines parsed
std::size_t parse(const std::string& data, std::istream& cache)
{
	std::size_t result = 0;

	// Detect script format then parse appropriately
	if (data.starts_with("BurikoCompiledScriptVer1.00\0"sv)) {
		std::uint32_t add_off{}, off{}, op{};
		if (!read_le(add_off, data, 0x1c))
			return false;
		off = (add_off += 0x1c); // Offset added to PUSH opcode string location, and first OP offset
		std::vector<std::string> stack;
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
			if (op == 0x140 && !stack.empty()) {
				// PRINT
				std::string name, text, textfix;
				text = std::move(stack.back());
				if (stack.size() > 1) {
					name = std::move(stack[0]);
				}

				if (stack.size() > 2)
					std::cerr << "PRINT: too big stack (parse error)" << std::endl;

				// Parse and register furigana
				static const std::regex rrgx("<R(.*?)>(.*?)</R>");
				auto words_begin = std::sregex_iterator(text.begin(), text.end(), rrgx);
				auto words_end = std::sregex_iterator();
				size_t shift = 0;
				textfix = text;

				for (std::sregex_iterator i = words_begin; i != words_end; i++) {
					auto read_as = i->str(1);
					auto type_as = i->str(2);

					textfix.replace(i->position() - shift, i->length(), type_as);
					shift += i->length() - type_as.length();

					// Don't register accent dots as furigana
					static constexpr auto rep_dot =
						"・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・・"sv;
					if (rep_dot.starts_with(read_as))
						continue;
					g_furigana.emplace(std::move(type_as), std::move(read_as));
				}

				if (textfix.find_first_of("<") + 1)
					std::cerr << "Unknown tag: " << textfix << std::endl;

				add_line(std::move(name), std::move(textfix), cache);
				result++;
			}

			stack.clear();
			continue;
		}

		return result;
	}

	return 0;
}
