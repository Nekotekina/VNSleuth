#include "parser_base.hpp"

using namespace std::literals;

std::unordered_multimap<std::string, std::function<parser_base&()>> parser_base::read_archive() const
{
	std::unordered_multimap<std::string, std::function<parser_base&()>> entries;
	if (data.starts_with("BURIKO ARC"sv)) {
		;
	}
	if (data.starts_with("PackFile\0"sv)) {
		;
	}

	return entries;
}
