#include "common.h"
#include "llama.h"
#include "main.hpp"
#include <chrono>
#include <deque>

extern volatile bool g_stop;

extern std::string example;
extern std::string iprefix;
extern std::string isuffix;

extern std::string print_line(line_id id, std::string* line);

extern void update_segment(std::string_view prompt, uint seg);

bool translate(gpt_params& params, line_id id)
{
	// Initialize llama.cpp
	static const auto [model, ctx] = [&]() -> std::tuple<llama_model*, llama_context*> {
		llama_backend_init();
		llama_numa_init(params.numa);
		return llama_init_from_gpt_params(params);
	}();

	static std::vector<llama_token> tokens; // Current tokens (not necessarily decoded)
	static std::deque<uint> chunks;			// Size (in tokens) of each translated message block
	static uint decoded = 0;				// Number of token successfully llama_decode()'d
	static uint segment = -1;				// Current segment

	static const bool _init = [&]() -> bool {
		if (!model || !ctx) {
			return false;
		}
		atexit([] {
			llama_free(ctx);
			llama_free_model(model);
			llama_backend_free();
		});

		return true;
	}();

	if (!_init) {
		std::cerr << "Failed to initialize llama model." << std::endl;
		return false;
	}

	static std::size_t sample_count = 0;
	static auto sample_time = 0ns;
	static std::size_t batch_count = 0;
	static auto batch_time = 0ns;
	static auto eval_time = 0ns;

	// Remove first message block (returns number of tokens to erase)
	auto eject_first = [&]() -> uint {
		if (chunks.empty()) {
			return 0;
		}

		auto count = chunks.front();
		chunks.pop_front();
		if (decoded > 0u + params.n_keep) {
			const auto p1 = params.n_keep + count;
			if (!llama_kv_cache_seq_rm(ctx, 0, params.n_keep, p1))
				throw std::runtime_error("llama_kv_cache_seq_rm 1");
			llama_kv_cache_seq_add(ctx, 0, p1, decoded, 0u - count);
			llama_kv_cache_defrag(ctx);
			llama_kv_cache_update(ctx);
			decoded -= count;
			if (decoded > 0u + params.n_ctx)
				throw std::out_of_range("ctx underflow");
			if (decoded < 0u + params.n_keep)
				throw std::out_of_range("eject_first failed");
		}
		return count;
	};

	// Remove last message block(s)
	auto eject_bunch = [&](uint i) {
		if (i >= chunks.size()) {
			// Remove all
			if (decoded) {
				// Print some statistics
				std::cerr << "Sample speed: ";
				std::cerr << uint(10 / (sample_time.count() / sample_count / 1e9)) / 10. << "/s" << std::endl;
				std::cerr << "Batch speed: ";
				std::cerr << uint(10 / (batch_time.count() / batch_count / 1e9)) / 10. << "/s (" << batch_count << " total)" << std::endl;
				std::cerr << "Eval speed: ";
				std::cerr << uint(10 / (eval_time.count() / sample_count / 1e9)) / 10. << "/s (" << sample_count << " total)" << std::endl;
			}

			chunks.clear();
			tokens.clear();
			decoded = 0;
			llama_kv_cache_clear(ctx);
			if (llama_add_bos_token(model)) {
				tokens.push_back(llama_token_bos(model));
			}

			for (auto& t : llama_tokenize(model, params.prompt + example, false)) {
				tokens.push_back(t);
			}

			params.n_keep = tokens.size();
			return;
		}
		while (i--) {
			auto count = chunks.back();
			chunks.pop_back();
			if (decoded == tokens.size()) {
				if (!llama_kv_cache_seq_rm(ctx, 0, decoded - count, -1))
					throw std::runtime_error("llama_kv_cache_seq_rm last");
				decoded -= count;
				if (decoded > tokens.size())
					throw std::out_of_range("eject_bunch");
				if (decoded <= 0u + params.n_keep)
					throw std::out_of_range("eject_bunch (unexpected)");
			}
			tokens.resize(tokens.size() - count);
		}
	};

	// Tokenize line and add to the tokens
	auto push_line = [&](const std::string& text) -> void {
		for (auto& t : llama_tokenize(model, text, false)) {
			tokens.push_back(t);
			chunks.back()++;
		}
	};

	auto init_segment = [&]() -> void {
		// Initialize segment: eject all first
		eject_bunch(-1);
		segment = id.first;
		if (segment + 1) {
			for (auto& line : g_lines.segs[segment].lines) {
				if (line.tr_text.empty())
					break;
				chunks.emplace_back();
				push_line(line.tr_text);
			}
		}
	};

	if (id.first != segment) {
		// Update previous translation file
		if (segment + 1) {
			update_segment(params.prompt, segment);
		}

		// Initialize new segment
		init_segment();
	}

	// Translate line(s)
	if (id != c_bad_id) {
		// Discard current and following lines
		uint to_eject = 0;
		for (line_id nid = id; nid != c_bad_id; g_lines.advance(nid)) {
			if (g_lines[nid].tr_text.empty())
				break;
			to_eject++;
			g_lines[nid].tr_text = {};
		}
		if (to_eject >= 10) {
			// Full reset
			init_segment();
		} else {
			eject_bunch(to_eject);
		}
	} else {
		return true;
	}

	// Detect additional lines for translation
	for (line_id pid{segment, 0}; pid <= id; g_lines.advance(pid)) {
		if (!g_lines[pid].tr_text.empty())
			continue;

		std::string llama_out;
		const auto spker = print_line(pid, &llama_out);
		chunks.emplace_back();
		if (auto seed = g_lines[pid].seed) {
			// Add dummy tokens to alter the output
			// Because simply changing seed almost never changes the output (because of low temp?)
			// TODO: explore other options, like removing one oldest chunk instead of adding tokens
			std::string dummy = iprefix;
			if (!dummy.ends_with(' '))
				dummy += ' ';
			dummy += std::to_string(seed);
			dummy += '\n';
			dummy += isuffix;
			if (!dummy.ends_with(' '))
				dummy += ' ';
			dummy += std::to_string(seed);
			dummy += '\n';
			push_line(dummy);
		}
		push_line(llama_out);

		// Eject old translations if necessary
		std::size_t count = 0;
		while (tokens.size() - count + params.n_predict > params.n_ctx - 1u) {
			if (auto c = eject_first()) {
				count += c;
			} else {
				std::cerr << "Prompt too big or context is too small" << std::endl;
				return false;
			}
		}

		tokens.erase(tokens.begin() + params.n_keep, tokens.begin() + (params.n_keep + count));

		// Decode pending tokens if necessary (TODO: split by batch size)
		if (auto bsize = tokens.size() - decoded) {
			auto stamp0 = std::chrono::steady_clock::now();
			llama_decode(ctx, llama_batch_get_one(&tokens[decoded], bsize, decoded, 0));
			llama_synchronize(ctx);
			auto stamp1 = std::chrono::steady_clock::now();
			batch_count += bsize;
			batch_time += stamp1 - stamp0;
			decoded = tokens.size();
		}

		static const auto sctx = [&]() {
			static const auto sctx = llama_sampling_init(params.sparams);
			atexit([]() { llama_sampling_free(sctx); });
			return sctx;
		}();
		llama_sampling_reset(sctx);
		llama_sampling_set_rng_seed(sctx, g_lines[pid].seed);
		static const auto real_isuffix = "\n" + isuffix;
		int pred_count = 0;
		while (!g_stop && !llama_out.ends_with("\n")) {
			// Predict next token
			auto stamp0 = std::chrono::steady_clock::now();
			auto token_id = llama_sampling_sample(sctx, ctx, nullptr);
			auto stamp1 = std::chrono::steady_clock::now();
			sample_count++;
			sample_time += stamp1 - stamp0;
			tokens.push_back(token_id);
			if (++pred_count > params.n_predict)
				token_id = llama_token_nl(model); // Force newline if size exceeded

			// TODO: support grammar
			llama_sampling_accept(sctx, ctx, token_id, false);
			auto token_str = llama_token_to_piece(ctx, token_id, true);

			auto trim = llama_out.ends_with(real_isuffix);
			llama_out += token_str;
			if (g_mode == op_mode::rt_llama) {
				if (trim && token_str.starts_with(" "))
					token_str.erase(0, 1);
				std::cout << token_str << std::flush; // Stream to terminal
			}

			stamp0 = std::chrono::steady_clock::now();
			if (auto result = llama_decode(ctx, llama_batch_get_one(&tokens[decoded], 1, decoded, 0))) {
				std::cerr << "llama_decode failed: " << result << std::endl;
				return false;
			}
			llama_synchronize(ctx);
			stamp1 = std::chrono::steady_clock::now();
			eval_time += stamp1 - stamp0;
			decoded++;
			if (chunks.empty())
				throw std::out_of_range("chunks.empty()");
			chunks.back()++;
			if (decoded > 0u + params.n_ctx)
				throw std::out_of_range("ctx overflow");
		}
		if (g_stop && !llama_out.ends_with("\n"))
			std::cout << std::endl; // Stop current line
		if (g_mode == op_mode::rt_llama)
			std::cout << "\033[0m" << std::flush; // Reset terminal colors
		if (g_stop)
			return true;

		// Try to parse translated name
		if (spker.size()) {
			std::string_view speaker = llama_out;
			if (auto pos = speaker.find_first_not_of(" ") + 1)
				speaker.remove_prefix(pos - 1);
			speaker = speaker.substr(0, speaker.find_first_of(':') + 1);
			if (speaker.empty() || speaker.find_first_of('"') + 1) {
				std::cerr << "Failed to parse speaker translation for: " << spker << std::endl;
				g_speakers[spker] = ":";
			} else {
				g_speakers[spker] = speaker;
			}
		}

		// Store translated line
		g_lines[pid].tr_text = std::move(llama_out);
	}

	return true;
}