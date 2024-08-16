#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

using ui64 = std::uint64_t;
using si64 = std::int64_t;

// Make basic type from anyvalue (int, uint = size of type in bytes)
template <auto V, typename E = void>
struct make_type {
	// Fallback, use value's type
	using type = decltype(V);
	static_assert(!std::is_integral_v<type>, "Unsupported integer size");
};

template <int V>
struct make_type<V, std::enable_if_t<(V > 0)>> {
	// Positive signed values result in fixed-sized string
	struct type {
		char data[V]{};

		type() = default;
		type(std::string_view str) { std::memcpy(+data, str.data(), std::min<std::size_t>(V, str.size())); }

		explicit operator bool() const { return data[0] != 0; }

		operator std::string_view() const
		{
			if (std::find(data + 0, data + V, '\0') == data + V)
				return std::string_view(data, V);
			else
				return data;
		}

		operator std::string() const
		{
			if (std::find(data + 0, data + V, '\0') == data + V)
				return std::string(data, V);
			else
				return data;
		}
	};
};
template <>
struct make_type<0> {
	// Dynamic-sized null-terminated string
	using type = std::string;
};
template <>
struct make_type<-1> {
	using type = signed char;
};
template <>
struct make_type<1u> {
	using type = unsigned char;
};
template <>
struct make_type<-2> {
	using type = short;
};
template <>
struct make_type<2u> {
	using type = unsigned short;
};
template <>
struct make_type<-4> {
	using type = int;
};
template <>
struct make_type<4u> {
	using type = unsigned int;
};
template <>
struct make_type<-8> {
	using type = std::int64_t;
};
template <>
struct make_type<8u> {
	using type = std::uint64_t;
};

template <auto V>
using make_type_t = typename make_type<V>::type;

using parser_func_t = std::function<const struct parser_base&(bool full, struct parser_base* opt_dst)>;

struct parser_base {
	const std::string_view data;

	explicit parser_base(std::string_view data) : data(data) {}

	// Read little-endian value
	template <typename T, typename Off>
	bool read_le(T& dst, Off&& pos) const noexcept
		requires(std::is_trivially_copyable_v<T>)
	{
		static_assert(std::endian::native == std::endian::little, "Big Endian platform support not implemented");
		if (std::size_t(pos) >= data.size() || std::size_t(pos) + sizeof(T) > data.size())
			return false;
		std::memcpy(&dst, data.data() + pos, sizeof(T));
		if constexpr (!std::is_const_v<std::remove_reference_t<Off>>)
			pos += sizeof(T); // Optionally increment position
		return true;
	}

	// Read null-terminated string (one byte after data shall be valid and zero, like in std::string)
	template <typename Off>
	bool read_le(std::string& dst, Off&& pos) const noexcept
	{
		if (std::size_t(pos) >= data.size())
			return false;
		dst = data.data() + pos;
		if constexpr (!std::is_const_v<std::remove_reference_t<Off>>)
			pos += data.size() + 1;
		return true;
	}

	// Helper
	template <typename... T, std::size_t... Idx>
	bool read_le(std::tuple<T..., bool>& dst, std::size_t& off, std::index_sequence<Idx...>) const noexcept
	{
		return (read_le(std::get<Idx>(dst), off) && ...);
	}

	// Read little-endian value(s) (as "packed" struct)
	template <typename... T, typename Off>
	std::tuple<T..., bool> read_le(Off&& pos) const noexcept
	{
		std::tuple<T..., bool> result{};
		std::size_t off = pos;
		if (read_le<T...>(result, off, std::make_index_sequence<sizeof...(T)>()))
			std::get<sizeof...(T)>(result) = true;
		if constexpr (!std::is_const_v<std::remove_reference_t<Off>>)
			pos += (off - pos); // Optionally increment position
		return result;
	}

	// Read little-endian value(s) with types encoded in template args
	template <auto... V, typename Off>
	std::tuple<make_type_t<V>..., bool> read_le(Off&& pos) const noexcept
	{
		std::tuple<make_type_t<V>..., bool> result{};
		std::size_t off = pos;
		if (read_le<make_type_t<V>...>(result, off, std::make_index_sequence<sizeof...(V)>()))
			std::get<sizeof...(V)>(result) = true;
		if constexpr (!std::is_const_v<std::remove_reference_t<Off>>)
			pos += (off - pos); // Optionally increment position
		return result;
	}

	template <typename Off>
	bool write_le(Off&&) noexcept
	{
		return true;
	}

	template <typename Off, typename T, typename... Ts>
	bool write_le(Off&& pos, const T& arg, const Ts&... args) noexcept
		requires(std::is_trivially_copyable_v<T>)
	{
		static_assert(std::endian::native == std::endian::little, "Big Endian platform support not implemented");
		std::size_t off = pos;
		if (off >= data.size() || off + sizeof(T) > data.size())
			return false;
		std::memmove(const_cast<char*>(data.data()) + off, &arg, sizeof(T));
		off += sizeof(T);
		if constexpr (!std::is_const_v<std::remove_reference_t<Off>>)
			pos = off; // Optionally increment position
		return write_le(std::forward<Off>(pos), args...);
	}

	template <typename Off, typename... Ts>
	bool write_le(Off&& pos, std::string_view str, const Ts&... args) noexcept
	{
		std::size_t off = pos;
		if (off >= data.size() || off + str.size() > data.size())
			return false;
		std::memmove(const_cast<char*>(data.data()) + off, str.data(), str.size());
		const_cast<char&>(data[off + str.size()]) = 0;
		off += str.size() + 1;
		if constexpr (!std::is_const_v<std::remove_reference_t<Off>>)
			pos = off; // Optionally increment position
		return write_le(std::forward<Off>(pos), args...);
	}

	template <auto... V, typename Off>
	bool write_le(Off&& pos, const make_type_t<V>&... args) noexcept
	{
		return write_le<Off, make_type_t<V>...>(std::forward<Off>(pos), args...);
	}

	// Copy bytes directly with overlapping technique (warning: will not be virtualized)
	bool copy_overlapped(std::size_t src_off, std::size_t dst_off, std::size_t count);

	// Parse container (full=false will return fake data with correct size, maybe some other info inside)
	// In the most simple case, function capture will contain single parser_base with a substring
	std::vector<std::pair<std::string, parser_func_t>> read_archive() const;
};

// Helper
struct owning_parser : private std::string, public parser_base {
	owning_parser(std::string&& data) : std::string(std::move(data)), parser_base(this->operator std::string_view()) {}
	owning_parser(std::size_t size) : std::string(size, '\0'), parser_base(this->operator std::string_view()) {}

	using parser_base::data;

	std::string move() { return std::move(*static_cast<std::string*>(this)); }
};

// clang-format off

// Base class for format-specific parsers
struct parser_base2 : protected parser_base {
	using parser_base::parser_base;

	// Usage:
	// auto [dst, skip] = get_dst(opt_dst, size, full, ".txt");
	// if (skip)
	//   return dst;
	std::pair<parser_base&, bool> get_dst(parser_base* opt_dst, std::size_t size, bool full, std::string_view fmt_id = "")
	{
		if (!full) {
			// Pack size and fmd_id
			m_inf.~owning_parser();
			new (&m_inf) owning_parser(fmt_id.size() + 8);
			m_inf.write_le(0, size);
			m_inf.write_le(8, fmt_id);
			return {m_inf, size + 1 != 0};
		} else if (opt_dst) {
			if ((size + 1 == 0 && !m_dst.data.empty()) || size == m_dst.data.size()) {
				// Copy existing data
				std::memcpy(const_cast<char*>(opt_dst->data.data()), m_dst.data.data(), std::min(m_dst.data.size(), opt_dst->data.size()));
				return {*opt_dst, true};
			} else {
				return {*opt_dst, false};
			}
		} else {
			if (size == m_dst.data.size()) {
				// Use existing data
				return {m_dst, true};
			} else {
				// If size == -1, it means size is unknown
				if (size + 1) {
					// Recreate
					m_dst.~owning_parser();
					new (&m_dst) owning_parser(size);
				}
				return {m_dst, false};
			}
		}
	}

	// For initially unknown size situation, reinitialize m_dst with prepared data
	parser_base& make_dst(parser_base* opt_dst, std::string data, bool full, std::string_view fmt_id = "")
	{
		m_dst.~owning_parser();
		new (&m_dst) owning_parser(std::move(data));
		if (!full) {
			// Pack size and fmd_id
			m_inf.~owning_parser();
			new (&m_inf) owning_parser(fmt_id.size() + 8);
			m_inf.write_le(0, m_dst.data.size());
			m_inf.write_le(8, fmt_id);
			return m_inf;
		} else if (opt_dst) {
			// Copy existing data
			std::memcpy(const_cast<char*>(opt_dst->data.data()), m_dst.data.data(), std::min(m_dst.data.size(), opt_dst->data.size()));
			return *opt_dst;
		} else {
			return m_dst;
		}
	}

	// For failure
	parser_base& null_dst(bool full)
	{
		m_dst.~owning_parser();
		new (&m_dst) owning_parser(0);
		if (!full) {
			m_inf.~owning_parser();
			new (&m_inf) owning_parser(8);
			return m_inf;
		} else {
			return m_dst;
		}
	};

	virtual parser_base& parse(bool full, parser_base* opt_dst) = 0;

private:
	owning_parser m_dst = std::string();
	owning_parser m_inf = std::string();
};

// For scattered specializations
template <typename T>
std::shared_ptr<parser_base2> make_format_parser(std::string_view data);

template <typename T>
parser_func_t make_lazy_format_parser(std::string_view data)
{
	return [parser = make_format_parser<T>(data)](bool full, parser_base* opt_dst) -> const parser_base& {
		return parser->parse(full, opt_dst);
	};
}

inline parser_func_t make_noop_parser(std::string_view data)
{
	return [parser = parser_base(data)](bool, parser_base*) -> const parser_base& {
		return parser;
	};
}
// clang-format on
