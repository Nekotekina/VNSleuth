#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <string_view>

struct bit_reader_base {
	std::string_view data;
	std::uint64_t pos = 0;

	bool is_eos(uint count = 1) const { return pos > data.size() * 8 || pos + count > data.size() * 8; }
};

struct bit_reader_lsb : bit_reader_base {
	uint get_bit()
	{
		if (is_eos())
			return 0;
		uint r = (data[pos / 8] >> (pos % 8)) & 1;
		pos++;
		return r;
	}

	std::uint64_t get_bits(uint count)
	{
		if (!count || is_eos(count)) [[unlikely]]
			return 0;
		if (count > 57) {
			// TODO: error
			pos += count - 57;
			count = 57;
		}
		std::uint64_t r = 0;
		if constexpr (false && std::endian::native == std::endian::little) {
			// First, copy as many bits as possible
			std::memcpy(&r, data.data() + pos / 8, 8);
			// Remove unnecessary bits
			r >>= (pos % 8);
			pos += count;
			r &= ((std::uint64_t(1) << count) - 1);
		} else {
			while (count--)
				r = (r << 1) | get_bit();
		}
		return r;
	}
};

struct bit_reader_msb : bit_reader_base {
	uint get_bit()
	{
		if (is_eos())
			return 0;
		uint r = (data[pos / 8] >> (~pos % 8)) & 1;
		pos++;
		return r;
	}

	std::uint64_t get_bits(uint count)
	{
		if (!count || is_eos(count)) [[unlikely]]
			return 0;
		if (count > 64) {
			pos += count - 64;
			count = 64;
		}
		std::uint64_t r = 0;
		while (count--)
			r = (r << 1) | get_bit();
		return r;
	}
};