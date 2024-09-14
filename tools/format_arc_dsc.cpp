//! \file       ArcBGI.cs
//! \date       Tue Sep 09 09:29:12 2014
//! \brief      BGI/Ethornell engine archive implementation.
//
// Copyright (C) 2014-2015 by morkt
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
#include "parser_bits.hpp"

struct parser_arc_dsc final : parser_base2 {
	using parser_base2::parser_base2;
	parser_base& parse(bool full, parser_base* opt_dst) override;
};

namespace {
// Used to "encrypt" Huffman table
struct MagicDecoder {
	uint m_key;
	uint m_magic;

	unsigned char UpdateKey()
	{
		uint v0 = 20021 * (m_key & 0xffff);
		uint v1 = m_magic | (m_key >> 16);
		v1 = v1 * 20021 + m_key * 346;
		v1 = (v1 + (v0 >> 16)) & 0xffff;
		m_key = (v1 << 16) + (v0 & 0xffff) + 1;
		return v1;
	}
};

struct alignas(uint) HuffmanCode {
	unsigned short Code;
	unsigned short Depth;

	bool operator<(const HuffmanCode& r) const
	{
		if (Depth == r.Depth)
			return Code < r.Code;
		return Depth < r.Depth;
	}
};

struct HuffmanNode {
	uint IsParent : 1 = 0;
	uint Code : 9 = 0;
	uint LeftChildIndex : 11 = 0;
	uint RightChildIndex : 11 = 0;
};
} // namespace

parser_base& parser_arc_dsc::parse(bool full, parser_base* opt_dst)
{
	auto [dst_size, dec_count, ok] = read_le<4u, 4u>(0x14);
	if (!ok || !data.starts_with("DSC FORMAT 1.00\0") || data.size() <= 512 + 0x20) [[unlikely]]
		return null_dst(full, "DSC format\n");
	auto [dst, skip] = get_dst(opt_dst, dst_size, full);
	if (skip)
		return dst;

	// Read and sort Huffman codes (512 bytes at offset 0x20)
	std::vector<HuffmanCode> hcodes{};
	hcodes.reserve(512);
	{
		MagicDecoder hcodes_dec{};
		read_le(hcodes_dec.m_magic, 0);
		hcodes_dec.m_magic <<= 16;
		read_le(hcodes_dec.m_key, 0x10);
		for (unsigned short i = 0; i < 512; i++) {
			unsigned char depth = data[0x20 + i];
			depth -= hcodes_dec.UpdateKey();
			if (depth)
				hcodes.emplace_back(HuffmanCode{i, depth});
		}
	}
	std::sort(hcodes.begin(), hcodes.end());

	// Generate Huffman nodes
	std::array<HuffmanNode, 1023> hnodes{};
	{
		std::array<int, 512> nodes_index[2];
		int next_node_index = 1;
		int depth_nodes = 1;
		int depth = 0;
		int child_index = 0;
		nodes_index[0][0] = 0;
		for (uint n = 0; n < hcodes.size();) {
			int huffman_nodes_index = child_index;
			child_index ^= 1;

			int depth_existed_nodes = 0;
			while (n < hcodes.size() && hcodes[n].Depth == depth) {
				HuffmanNode node{.IsParent = 0, .Code = hcodes[n++].Code};
				hnodes[nodes_index[huffman_nodes_index][depth_existed_nodes]] = node;
				depth_existed_nodes++;
			}
			int depth_nodes_to_create = depth_nodes - depth_existed_nodes;
			for (int i = 0; i < depth_nodes_to_create; i++) {
				HuffmanNode node{.IsParent = 1};
				nodes_index[child_index][i * 2] = node.LeftChildIndex = next_node_index++;
				nodes_index[child_index][i * 2 + 1] = node.RightChildIndex = next_node_index++;
				hnodes[nodes_index[huffman_nodes_index][depth_existed_nodes + i]] = node;
			}
			depth++;
			depth_nodes = depth_nodes_to_create * 2;
		}
	}

	// Decompress
	bit_reader_msb bits{data.substr(512 + 0x20)};
	{
		uint dst_ptr = 0;
		for (uint k = 0; k < dec_count; k++) {
			uint node_index = 0;
			do {
				if (bits.is_eos()) [[unlikely]]
					return null_dst(full, "bits.is_eos\n");
				uint bit = bits.get_bit();
				if (0 == bit)
					node_index = hnodes[node_index].LeftChildIndex;
				else
					node_index = hnodes[node_index].RightChildIndex;
			} while (hnodes[node_index].IsParent);

			uint code = hnodes[node_index].Code;
			if (code >= 256) {
				if (bits.is_eos(12)) [[unlikely]]
					return null_dst(full, "bits.is_eos\n");
				uint offset = bits.get_bits(12) + 2;
				uint count = (code & 0xff) + 2;
				if (dst_ptr < offset || dst_ptr + count > dst_size) [[unlikely]]
					return null_dst(full, "dst_ptr\n");
				if (!dst.copy_overlapped(dst_ptr - offset, dst_ptr, count))
					return null_dst(full, "copy_overlapped'n");
				dst_ptr += count;
			} else {
				if (!dst.write_le<1u>(dst_ptr, code)) [[unlikely]]
					return null_dst(full, "write\n");
			}
		}
	}

	return dst;
}

template <>
std::shared_ptr<parser_base2> make_format_parser<parser_arc_dsc>(std::string_view data)
{
	return std::make_shared<parser_arc_dsc>(data);
}
