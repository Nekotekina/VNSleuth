#include "common.h"
#include "llama.h"
#include "main.hpp"
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>

extern volatile bool g_stop;

extern std::string example;
extern std::string iprefix;
extern std::string isuffix;

extern std::string print_line(line_id id, std::string* line, bool stream);

extern void update_segment(std::string_view prompt, uint seg);

bool translate(gpt_params& params, line_id id, tr_cmd cmd)
{
	static const auto s_main_tid = std::this_thread::get_id();
	static std::atomic<uint> stop_sig = -1; // stop signal for thread, new id.second is sent
	static std::atomic<uint> work_res = 0;	// number of translated lines in segment, done by worker
	static std::thread worker;

	static auto is_stopped = []() -> bool {
		if (g_stop)
			return true;
		return std::this_thread::get_id() != s_main_tid && stop_sig + 1;
	};

	static auto join_worker = [](uint val) {
		if (worker.joinable()) {
			stop_sig = val;
			stop_sig.notify_all();
			worker.join();
		}
	};

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

	static const struct _init_t {
		explicit operator bool() const { return model && ctx; }
		_init_t() = default;
		~_init_t()
		{
			join_worker(0);
			llama_free(ctx);
			llama_free_model(model);
			llama_backend_free();
		}
	} _init{};

	if (!_init) {
		std::cerr << "Failed to initialize llama model." << std::endl;
		return false;
	}

	static std::size_t sample_count = 0;
	static auto sample_time = 0ns;
	static std::size_t batch_count = 0;
	static auto batch_time = 0ns;
	static auto eval_time = 0ns;

	if (cmd == tr_cmd::sync) {
		// Abort background worker and possibly discard work
		// Send Id to start discarding from, obviously can't use -1
		join_worker(std::min<uint>(id.second, -2));
		return true;
	}

	if (std::this_thread::get_id() == s_main_tid) {
		// Stop background worker first
		if (cmd == tr_cmd::kick && segment == id.first && worker.joinable()) {
			// Kick only once last translated line has been read
			uint tr_lines = work_res;
			uint rd_lines = id.second + 1;
			// Compare translated count with the number of read lines
			if (tr_lines > rd_lines) {
				// Don't touch worker
				return true;
			} else if (tr_lines != rd_lines) {
				std::fprintf(stderr, "\033[0mError: Kicked from untranslated line: %u<%u\n", tr_lines, rd_lines);
			}
		}

		if (cmd == tr_cmd::kick || cmd == tr_cmd::translate) {
			join_worker(id.first != segment ? 0 : id.second + 1);
		}
	}

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
		if (i > chunks.size()) {
			// Remove all
			if (decoded && batch_count && sample_count) {
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
				if (decoded > tokens.size() || decoded < 0u + params.n_keep)
					throw std::out_of_range("eject_bunch decoded=" + std::to_string(decoded));
			}
			tokens.resize(tokens.size() - count);
		}
	};

	// Eject old translations if necessary keeping only 75% of the context full (TODO)
	auto eject_start = [&]() -> bool {
		std::size_t count = 0;
		while (tokens.size() - count + params.n_predict > params.n_ctx * 3 / 4 - 1u) {
			if (auto c = eject_first()) {
				count += c;
			} else {
				std::cerr << "Prompt too big or context is too small" << std::endl;
				return false;
			}
		}

		tokens.erase(tokens.begin() + params.n_keep, tokens.begin() + (params.n_keep + count));
		return true;
	};

	// Tokenize line and add to the tokens
	auto push_line = [&](const std::string& text) -> uint {
		uint r = 0;
		for (auto& t : llama_tokenize(model, text, false)) {
			tokens.push_back(t);
			chunks.back()++;
			r++;
		}
		return r;
	};

	auto decode = [&]() -> void {
		// (TODO: split by batch size?)
		if (auto bsize = tokens.size() - decoded) {
			auto stamp0 = std::chrono::steady_clock::now();
			llama_decode(ctx, llama_batch_get_one(&tokens[decoded], bsize, decoded, 0));
			llama_synchronize(ctx);
			auto stamp1 = std::chrono::steady_clock::now();
			batch_count += bsize;
			batch_time += stamp1 - stamp0;
			decoded = tokens.size();
		}
	};

	auto init_segment = [&]() -> void {
		// Initialize segment: eject all first
		eject_bunch(-1);
		decode();
		if (segment + 1) {
			std::lock_guard lock(g_mutex);

			for (auto& line : g_lines.segs[segment].lines) {
				if (line.tr_text.empty())
					break;
				chunks.emplace_back();
				push_line(line.tr_text);
			}
		}
	};

	if (cmd == tr_cmd::eject) {
		eject_bunch(id.second);
		return true;
	}

	if (cmd == tr_cmd::reload) {
		if (id == c_bad_id) {
			// Full reload
			init_segment();
			return true;
		}
		if (id.first != segment) {
			segment = id.first;
			init_segment();
		}
		if (segment + 1) {
			std::lock_guard lock(g_mutex);

			for (line_id nid{segment, id.second}; nid != c_bad_id; g_lines.advance(nid)) {
				if (g_lines[nid].tr_text.empty())
					break;
				chunks.emplace_back();
				push_line(g_lines[nid].tr_text);
			}
		}
		decode();
		return true;
	}

	if (id.first != segment) {
		// Update previous translation file
		if (segment + 1) {
			update_segment(params.prompt, segment);
		}

		// Initialize new segment
		segment = id.first;
		init_segment();
	}

	// Translate line(s)
	if (id != c_bad_id) {
		// Discard current and following lines
		uint to_eject = 0;
		for (line_id nid = id; cmd == tr_cmd::translate && nid != c_bad_id; g_lines.advance(nid)) {
			std::lock_guard lock(g_mutex);

			if (g_lines[nid].tr_text.empty())
				break;
			to_eject++;
			if (params.verbosity)
				std::fprintf(stderr, "\033[0m[id:%u:%u] Ejected\n", nid.first, nid.second);
			g_lines[nid].tr_text = {};
			if (nid > id)
				g_lines[nid].seed = 0;
		}
		// TODO: more accurate computation with accounting for context size
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
	for (line_id pid{segment, 0}; cmd == tr_cmd::translate && pid <= id; g_lines.advance(pid)) {
		uint seed = g_lines[pid].seed;
		if (!(std::lock_guard{g_mutex}, g_lines[pid].tr_text.empty()))
			continue;

		if (std::this_thread::get_id() != s_main_tid) {
			if (g_lines[pid].seed) {
				std::cerr << "Unexpected: seed not reset" << std::endl;
			}
			// Don't eject context from worker thread; instead, allow filling full context in it
			if (tokens.size() + params.n_predict > params.n_ctx - 1u) {
				// Nothing to do for worker thread here
				return false;
			}
		}

		if (params.verbosity)
			std::fprintf(stderr, "\033[0m[id:%u:%u, s:%u, t:%zu, last:%u, chk:%llu] Line: %s%s\n", pid.first, pid.second, seed, tokens.size(),
						 chunks.size() ? chunks.back() : -1, std::accumulate(tokens.begin(), tokens.end(), 0ull), g_lines[pid].name.c_str(),
						 g_lines[pid].text.data());

		std::string llama_out;
		const auto spker = print_line(pid, &llama_out, std::this_thread::get_id() == s_main_tid);
		chunks.emplace_back();
		uint dummy_size = 0;
		if (seed && std::this_thread::get_id() == s_main_tid) {
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
			dummy_size = push_line(dummy);
		}
		push_line(llama_out);
		// Worker thread doesn't eject translations
		if (std::this_thread::get_id() == s_main_tid && !eject_start())
			return false;

		// Decode pending tokens if necessary
		decode();

		// Eject dummy immediately (optimization)
		if (dummy_size) {
			const auto p0 = decoded - chunks.back();
			if (!llama_kv_cache_seq_rm(ctx, 0, p0, p0 + dummy_size))
				throw std::runtime_error("llama_kv_cache_seq_rm dummy");
			llama_kv_cache_seq_add(ctx, 0, p0 + dummy_size, decoded, 0u - dummy_size);
			tokens.erase(tokens.begin() + p0, tokens.begin() + (p0 + dummy_size));
			chunks.back() -= dummy_size;
			decoded -= dummy_size;
		}

		static const auto sctx = [&]() {
			static const auto sctx = llama_sampling_init(params.sparams);
			atexit([]() { llama_sampling_free(sctx); });
			return sctx;
		}();
		llama_sampling_reset(sctx);
		llama_sampling_set_rng_seed(sctx, seed);
		static const auto real_isuffix = "\n" + isuffix;
		int pred_count = 0;
		while (!is_stopped() && !llama_out.ends_with("\n")) {
			// Predict next token
			auto stamp0 = std::chrono::steady_clock::now();
			auto token_id = llama_sampling_sample(sctx, ctx, nullptr);
			auto stamp1 = std::chrono::steady_clock::now();
			sample_count++;
			sample_time += stamp1 - stamp0;
			if (++pred_count > params.n_predict)
				token_id = llama_token_nl(model); // Force newline if size exceeded
			tokens.push_back(token_id);

			// TODO: support grammar
			llama_sampling_accept(sctx, ctx, token_id, false);
			auto token_str = llama_token_to_piece(ctx, token_id, true);

			auto trim = llama_out.ends_with(real_isuffix);
			llama_out += token_str;
			if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid) {
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
		if (is_stopped() && !llama_out.ends_with("\n") && std::this_thread::get_id() == s_main_tid)
			std::cout << std::endl; // Stop current line
		if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid)
			std::cout << "\033[0m" << std::flush; // Reset terminal colors
		if (is_stopped() && !llama_out.ends_with("\n")) {
			// Discard incomplete work
			eject_bunch(1);
			return true;
		}

		// Try to parse translated name
		std::lock_guard lock(g_mutex);

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

	if (!is_stopped() && std::this_thread::get_id() == s_main_tid && g_mode == op_mode::rt_llama && id != c_bad_id) {
		// Launch worker thread to continue translation in the background
		// Check if the next line is untranslated
		if (auto next = g_lines.next(id); next != c_bad_id) {
			std::lock_guard lock(g_mutex);
			if (!g_lines[next].tr_text.empty())
				return true;
			g_lines[next].seed = 0;
		} else {
			// Nothing to do
			return true;
		}

		// Make some space
		if (!eject_start())
			return false;

		stop_sig = -1;
		work_res = id.second + 1;
		worker = std::thread([=, &params] {
			auto start = g_lines.next(id);
			auto next = start;
			if (params.verbosity)
				std::fprintf(stderr, "\033[0m[id:%u:%u] Thread entry\n", next.first, next.second);
			while (!is_stopped() && next != c_bad_id && translate(params, next, tr_cmd::translate)) {
				work_res++;
				g_lines.advance(next);
			}
			// Wait for thread's destiny instead of terminating it immediately
			while (stop_sig == 0u - 1)
				stop_sig.wait(-1);
			// Discard translations which were done in the background
			std::lock_guard lock(g_mutex);

			bool msg = false;
			uint from = stop_sig.load();
			while (start != c_bad_id && !g_lines[start].tr_text.empty()) {
				if (start.second >= from) {
					if (params.verbosity && !std::exchange(msg, true))
						std::fprintf(stderr, "\033[0m[id:%u:%u, from:%u] Cleaning\n", start.first, start.second, from);
					g_lines[start].tr_text.clear();
					eject_bunch(1); // reverse order but should work ok?
				}
				g_lines.advance(start);
			}
			if (params.verbosity)
				std::fprintf(stderr, "\033[0m[id:%u:?] Thread exit\n", next.first);
		});
	}

	return true;
}
