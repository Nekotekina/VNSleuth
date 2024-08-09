#pragma once

#include <bit>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

// Make basic type from anyvalue (int, uint = size of type in bytes)
template <auto V, typename E = void>
struct make_type {
	// Fallback, use value's type
	using type = decltype(V);
	static_assert(!std::is_integral_v<type>, "Unsupported integer size");
};

template <int V>
struct make_type<V, std::enable_if_t<(V > 1)>> {
	// Positive signed values result in fixed-sized string
	struct char_array {
		char data[V];

		explicit operator bool() const { return data[0] != 0; }

		operator std::string_view() const
		{
			if (std::find(data + 0, data + V, '\0') == data + V)
				return data;
			else
				return std::string_view(data, V);
		}

		operator std::string() const
		{
			if (std::find(data + 0, data + V, '\0') == data + V)
				return data;
			else
				return std::string(data, V);
		}
	};
};
template <>
struct make_type<0> {
	// Dynamic-sized null-terminated string
	using type = std::string_view;
};
template <>
struct make_type<1> {
	using type = char;
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
	using type = long long;
};
template <>
struct make_type<8u> {
	using type = unsigned long long;
};

template <auto V>
using make_type_t = typename make_type<V>::type;

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
	bool read_le(std::string_view& dst, Off&& pos) const noexcept
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

	// Parse container
	// In the most simple case, function capture will contain single parser_base with a substring
	std::unordered_multimap<std::string, std::function<parser_base&()>> read_archive() const;
};
