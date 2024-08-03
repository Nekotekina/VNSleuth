#pragma once

#include "tools/parser_base.hpp"

struct script_parser : private parser_base {
	// Publish members and constructors
	using parser_base::data;
	using parser_base::parser_base;

	// Recursively parse script data
	void read_segments(const std::string& name);
};
