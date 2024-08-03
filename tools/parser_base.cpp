//
// Copyright (C) 2014-2015 by morkt (C#)
// Copyright (C) 2024 by Nekotekina (C++ port)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//

#include "parser_base.hpp"
#include <cstdint>
#include <iostream>

using namespace std::literals;

// Copy potentially overlapping sequence of `count` bytes from `src` position to `dst`.
// If destination offset resides within source region then sequence will repeat itself.
// Widely used in various compression techniques.
bool parser_base::copy_overlapped(std::size_t src, std::size_t dst, std::size_t count)
{
	if (src > data.size() || src + count > data.size())
		return false;
	if (dst > data.size() || dst + count > data.size())
		return false;

	if (dst > src) {
		while (count > 0) {
			std::size_t preceding = std::min(dst - src, count);
			std::memmove(const_cast<char*>(data.data() + dst), data.data() + src, preceding);
			dst += preceding;
			count -= preceding;
		}
	} else {
		std::memmove(const_cast<char*>(data.data() + dst), data.data() + src, count);
	}

	return true;
}

struct parser_arc_dsc;

std::vector<std::pair<std::string, parser_func_t>> parser_base::read_archive() const
{
	std::vector<std::pair<std::string, parser_func_t>> result;
	if (data.starts_with("BURIKO ARC20"sv)) {
		// BGI/Ethornell engine archive v2
		auto [count, ok] = read_le<4u>(0xc);
		ui64 index_offset = 0x10;
		ui64 start_offset = count * 0x80ull + index_offset;
		if (ok && start_offset < data.size()) {
			for (uint i = 0; i < count; i++) {
				auto [name, off, size, pad_, ok] = read_le<0x60, 4u, 4u, 0x18>(index_offset);
				if (ok && off < data.size() && off + start_offset < data.size()) {
					auto subs = data.substr(off + start_offset, size);
					if (subs.size() != size)
						continue;
					if (size > 0x220 && subs.starts_with("DSC FORMAT 1.00\0"))
						result.emplace_back(name, make_lazy_format_parser<parser_arc_dsc>(subs));
					else
						result.emplace_back(name, make_noop_parser(subs));
				}
			}
		}
	}
	if (data.starts_with("PackFile    "sv)) {
		// BGI/Ethornell engine archive v1
		auto [count, ok] = read_le<4u>(0xc);
		ui64 index_offset = 0x10;
		ui64 start_offset = count * 0x20ull + index_offset;
		if (ok && start_offset < data.size()) {
			for (uint i = 0; i < count; i++) {
				auto [name, off, size, pad_, ok] = read_le<0x10, 4u, 4u, 8>(index_offset);
				if (ok && off < data.size() && off + start_offset < data.size()) {
					auto subs = data.substr(off + start_offset, size);
					if (subs.size() != size)
						continue;
					if (size > 0x220 && subs.starts_with("DSC FORMAT 1.00\0"))
						result.emplace_back(name, make_lazy_format_parser<parser_arc_dsc>(subs));
					else
						result.emplace_back(name, make_noop_parser(subs));
				}
			}
		}
	}

	return result;
}
