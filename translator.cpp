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

extern void update_segment(uint seg, bool upd_names);

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
	static std::string* prev_tail{};

	static const struct _init_t {
		explicit operator bool() const { return model && ctx; }
		_init_t(gpt_params& params)
		{
			if (!model || !ctx || params.n_draft <= 0)
				return;
		}
		~_init_t()
		{
			join_worker(0);
			llama_free(ctx);
			llama_free_model(model);
			llama_backend_free();
		}
	} _init{params};

	if (!_init) {
		std::cerr << "Failed to initialize llama model." << std::endl;
		return false;
	}

	static std::uint64_t sample_count = 0;
	static auto sample_time = 0ns;
	static std::uint64_t batch_count = 0;
	static auto batch_time = 0ns;
	static auto eval_time = 0ns;

	// Print some statistics
	auto print_stats = [&]() {
		if (decoded && batch_count && sample_count) {
			std::cerr << "Sample speed: ";
			std::cerr << uint(10 / (sample_time.count() / sample_count / 1e9)) / 10. << "/s" << std::endl;
			std::cerr << "Batch speed: ";
			std::cerr << uint(10 / (batch_time.count() / batch_count / 1e9)) / 10. << "/s (" << batch_count << " total)" << std::endl;
			std::cerr << "Eval speed: ";
			std::cerr << uint(10 / (eval_time.count() / sample_count / 1e9)) / 10. << "/s (" << sample_count << " total)" << std::endl;
		}
	};

	if (cmd == tr_cmd::sync) {
		// Abort background worker and possibly discard work
		// Send Id to start discarding from, obviously can't use -1
		join_worker(std::min<uint>(id.second, -2));
		if (is_stopped())
			print_stats();
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
				std::fprintf(stderr, "%sError: Kicked from untranslated line: %u<%u\n", g_esc.reset, tr_lines, rd_lines);
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
			if (chunks.empty()) {
				std::cerr << "Prompt too big or context is too small" << std::endl;
				return false;
			}
			count += eject_first();
		}

		tokens.erase(tokens.begin() + params.n_keep, tokens.begin() + (params.n_keep + count));
		return true;
	};

	// Tokenize line and add to the tokens
	auto push_str = [&](const std::string& text) -> uint {
		uint r = 0;
		auto tt = llama_tokenize(model, text, false);
		if (params.verbosity && !tt.empty())
			std::fprintf(stderr, "%s[tokens:%zu] push %zu tokens: '%s'\n", g_esc.reset, tokens.size(), tt.size(), text.c_str());
		for (auto& t : tt) {
			tokens.push_back(t);
			chunks.back()++;
			r++;
		}
		return r;
	};

	// Tokenize tr_text from id
	auto push_id = [&](line_id id) {
		auto tr_text = std::string_view(g_lines[id].tr_text);
		auto spos = tr_text.find("\n" + isuffix) + 1;
		if (!spos) {
			throw std::runtime_error("Line untranslated: " + g_lines[id].tr_text);
		}
		// Tokenize original strings with annotations
		std::size_t pref_pos = 0, post_pos = 0;
		if (!tr_text.starts_with(iprefix)) {
			post_pos = pref_pos = tr_text.find("\n" + iprefix) + 1;
			if (pref_pos == 0)
				throw std::runtime_error("Line without iprefix: " + g_lines[id].tr_text);
		}
		// Replay print_line logic
		auto out = apply_replaces(g_lines[id].text);
		// TODO: implement as regex replace
		if (out.starts_with("…") && out != "…")
			out.erase(0, "…"sv.size());
		// Pre-annotations: no replaces, then original line with replaces
		out = std::string(tr_text.substr(0, pref_pos)) + iprefix + g_lines[id].name + std::move(out) + "\n";
		// Post-annotations: no replaces
		post_pos += iprefix.size();
		post_pos += g_lines[id].name.size();
		post_pos += g_lines[id].text.size() + 1;
		out += tr_text.substr(post_pos, spos - post_pos);
		out += isuffix;
		push_str(out);
		tr_text.remove_prefix(spos);
		tr_text.remove_prefix(isuffix.size());
		int token_count = 0;
		if (!g_lines[id].name.empty()) {
			// Tokenize name separately: find ": " delimiter
			auto pos = tr_text.find(": ") + 1;
			if (!pos) {
				throw std::runtime_error("Name untranslated: " + g_lines[id].tr_text);
			}
			if (isuffix.ends_with(" "))
				pos++;
			token_count += push_str(std::string(tr_text.substr(0, pos)));
			tr_text.remove_prefix(pos);
		}
		token_count += push_str(std::string(tr_text));
		g_lines[id].tr_tts.assign(tokens.end() - token_count - 1, tokens.end());
		if (token_count > params.n_predict + 1)
			throw std::runtime_error("Line too long: " + g_lines[id].tr_text);
	};

	auto decode = [&]() -> void {
		auto stamp0 = std::chrono::steady_clock::now();
		uint total = 0;
		while (uint bsize = std::min<uint>(tokens.size() - decoded, params.n_batch)) {
			if (is_stopped())
				break;
			llama_decode(ctx, llama_batch_get_one(&tokens[decoded], bsize, decoded, 0));
			decoded += bsize;
			total += bsize;
		}
		llama_synchronize(ctx);
		auto stamp1 = std::chrono::steady_clock::now();
		if (total) {
			batch_count += total;
			batch_time += stamp1 - stamp0;
		}
	};

	auto init_segment = [&]() -> void {
		// Initialize segment: eject all first
		print_stats();
		eject_bunch(-1);

		// Load full history
		prev_tail = nullptr;
		for (auto& id : g_history) {
			if (g_lines[id].tr_text.empty())
				break;
			if (id.second == 0 && prev_tail) {
				chunks.emplace_back();
				push_str(*prev_tail);
				prev_tail = nullptr;
			}
			chunks.emplace_back();
			push_id(id);
			auto& tail = g_lines.segs[id.first].tr_tail;
			if (g_lines.is_last(id)) {
				prev_tail = &tail;
			}
		}
	};

	auto add_tail_finalize = [&]() -> void {
		if (segment + 1) {
			auto& tail = g_lines.segs[segment].tr_tail;
			if (id.second == 0 && !g_lines.segs[segment].lines.back().tr_text.empty()) {
				// Finalize segment if necessary
				if (tail.empty()) {
					tail = "\n";
					update_segment(segment, true);
				}
				prev_tail = nullptr;
				chunks.emplace_back();
				push_str(tail);
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
			add_tail_finalize();
			segment = id.first;
		}
		if (segment + 1) {
			for (line_id nid{segment, id.second}; g_lines.is_valid(nid); g_lines.advance(nid)) {
				if (g_lines[nid].tr_text.empty())
					break;
				chunks.emplace_back();
				push_id(nid);
			}
		}
		decode();
		return true;
	}

	if (id.first != segment) {
		// Update previous translation file
		if (segment + 1) {
			add_tail_finalize();
		}

		segment = id.first;
		if (tokens.empty())
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
				std::fprintf(stderr, "%s[id:%u:%u] Ejected\n", g_esc.reset, nid.first, nid.second);
			g_lines[nid].tr_text = {};
			if (nid > id) {
				g_lines[nid].seed = 0;
				g_lines[nid].tr_tts.clear();
			}
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
		auto& line = g_lines[pid];
		uint seed = line.seed;
		if (!(std::lock_guard{g_mutex}, line.tr_text.empty()))
			continue;

		std::string llama_out = prev_tail && pid.second == 0 ? *prev_tail : std::string();
		prev_tail = nullptr;
		// TODO: this is broken as annotations should be preserved on rewrites
		llama_out += std::exchange(g_lines.segs[pid.first].tr_tail, "");
		std::string spker = print_line(pid, &llama_out, std::this_thread::get_id() == s_main_tid);
		chunks.emplace_back();
		uint dummy_size = 0;
		// For some reason, seed 1 often produces more correct output. So why not always add a "dummy" (TODO)
		if (true || seed) {
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
			dummy_size = push_str(dummy);
		}
		push_str(llama_out);
		// Encode speaker separately and count its tokens
		int pred_count = push_str(spker);
		llama_out = spker;
		if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid) {
			std::cout << g_esc.tran << spker.c_str() + spker.starts_with(" ");
			std::cout << std::flush; // Stream to terminal
		}
		// Worker thread doesn't eject translations
		if (std::this_thread::get_id() == s_main_tid && !eject_start())
			return false;
		if (std::this_thread::get_id() != s_main_tid) {
			// Don't eject context from worker thread; instead, allow filling full context in it
			if (tokens.size() + params.n_predict > params.n_ctx - 1u) {
				// Nothing to do for worker thread here
				tokens.resize(tokens.size() - chunks.back());
				chunks.pop_back();
				return false;
			}
		}

		if (params.verbosity)
			std::fprintf(stderr, "%s[id:%u:%u, s:%u, t:%zu, last:%u, chk:%llu] Line: %s%s\n", g_esc.reset, pid.first, pid.second, seed, tokens.size(),
						 chunks.size() ? chunks.back() : -1, std::accumulate(tokens.begin(), tokens.end(), 0ull), line.name.c_str(),
						 line.text.data());

		// Decode pending tokens if necessary
		decode();

		static const auto sctx = [&]() {
			static const auto sctx = llama_sampling_init(params.sparams);
			atexit([]() { llama_sampling_free(sctx); });
			return sctx;
		}();
		llama_sampling_reset(sctx);
		llama_sampling_set_rng_seed(sctx, seed);
		const bool use_grammar = !line.name.empty();
		for (int i = 0; i < pred_count; i++)
			llama_sampling_accept(sctx, ctx, *(tokens.rbegin() + pred_count - i), use_grammar);
		static const auto real_isuffix = "\n" + isuffix;
		while (!is_stopped() && !llama_out.ends_with("\n")) {
			// Predict next token
			auto stamp0 = std::chrono::steady_clock::now();
			auto token_id = llama_sampling_sample(sctx, ctx, nullptr);
			auto stamp1 = std::chrono::steady_clock::now();
			sample_count++;
			sample_time += stamp1 - stamp0;
			if (++pred_count > params.n_predict)
				token_id = llama_token_nl(model); // Force newline if size exceeded
			if (llama_token_is_eog(model, token_id))
				token_id = llama_token_nl(model); // Force newline on EOT/EOS
			auto token_str = llama_token_to_piece(ctx, token_id, true);
			if (std::count(token_str.begin(), token_str.end(), '\n') > 0u + token_str.ends_with("\n")) {
				// Attempt to fix the line containing incorrect newlines
				token_str.resize(token_str.find_first_of('\n') + 1);
				auto tks = llama_tokenize(model, token_str, false);
				if (tks.size() == 1)
					token_id = tks[0];
				else
					token_id = llama_token_nl(model);
			}
			tokens.push_back(token_id);

			llama_sampling_accept(sctx, ctx, token_id, use_grammar);
			llama_out += token_str;
			int to_decode = 1;

			// Try to parse translated name
			if (spker.empty() && !line.name.empty()) {
				std::string_view speaker = llama_out;
				if (auto pos = speaker.find_first_not_of(" ") + 1)
					speaker.remove_prefix(pos - 1);
				speaker = speaker.substr(0, speaker.find(":") + 1);
				if (!speaker.empty()) {
					std::lock_guard lock(g_mutex);
					spker = g_speakers[line.name] = speaker;
				}
			}
			if (!spker.empty() || line.name.empty()) {
				// Perform limited replaces (only repeatable ones: TODO)
				auto fixed = apply_replaces(llama_out, true, spker.size());
				if (!fixed.empty() && fixed != llama_out) {
					if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid) {
						// Overwrite current line
						std::cout << std::string(llama_out.size(), '\b') << g_esc.tran << fixed.c_str() + (fixed.starts_with(" "));
						std::cout << std::flush;
					}
					// Retokenize and replace only the few last tokens
					auto tt = llama_tokenize(model, fixed, false);
					auto [mis, mistt] = std::mismatch(tokens.end() - pred_count, tokens.end(), tt.begin(), tt.end());
					uint miscount = tokens.end() - mis;
					int dropped = mis != tokens.end();
					tokens.erase(mis, tokens.end());
					int to_remove = miscount - dropped;
					to_decode += tt.end() - mistt;
					to_decode -= dropped;
					tokens.insert(tokens.end(), mistt, tt.end());
					chunks.back() -= to_remove;
					const auto p0 = decoded - to_remove;
					decoded -= to_remove;
					if (!llama_kv_cache_seq_rm(ctx, 0, p0, p0 + to_remove))
						throw std::runtime_error("llama_kv_cache_seq_rm replaces");
					llama_sampling_reset(sctx);
					for (auto& t : tt)
						llama_sampling_accept(sctx, ctx, t, use_grammar);
					llama_out = std::move(fixed);
					pred_count += to_decode - to_remove - 1;
					token_str.clear();
				}
			}
			if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid && !token_str.empty()) {
				std::cout << token_str.c_str() + (token_str.starts_with(" ") && llama_out == token_str);
				std::cout << std::flush; // Stream to terminal
			}

			stamp0 = std::chrono::steady_clock::now();
			if (auto result = llama_decode(ctx, llama_batch_get_one(&tokens[decoded], to_decode, decoded, 0))) {
				std::cerr << "llama_decode failed: " << result << std::endl;
				return false;
			}
			llama_synchronize(ctx);
			stamp1 = std::chrono::steady_clock::now();
			eval_time += stamp1 - stamp0;
			decoded += to_decode;
			if (chunks.empty())
				throw std::out_of_range("chunks.empty()");
			chunks.back() += to_decode;
			if (decoded > 0u + params.n_ctx)
				throw std::out_of_range("ctx overflow");
		}
		if (is_stopped() && !llama_out.ends_with("\n") && std::this_thread::get_id() == s_main_tid)
			std::cout << std::endl; // Stop current line
		if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid)
			std::cout << g_esc.reset << std::flush; // Reset terminal colors
		if (is_stopped() && !llama_out.ends_with("\n")) {
			// Discard incomplete work
			eject_bunch(1);
			return true;
		}
		if (dummy_size) {
			// Eject dummy
			const auto p0 = decoded - chunks.back();
			if (!llama_kv_cache_seq_rm(ctx, 0, p0, p0 + dummy_size))
				throw std::runtime_error("llama_kv_cache_seq_rm dummy");
			llama_kv_cache_seq_add(ctx, 0, p0 + dummy_size, decoded, 0u - dummy_size);
			tokens.erase(tokens.begin() + p0, tokens.begin() + (p0 + dummy_size));
			chunks.back() -= dummy_size;
			decoded -= dummy_size;
		}

		if (spker.empty() && !line.name.empty())
			std::cerr << "Failed to parse speaker translation for: " << line.name << std::endl;

		// Store tokens
		line.tr_tts.assign(tokens.end() - pred_count - 1, tokens.end());

		// Store translated line
		std::lock_guard lock(g_mutex);
		line.tr_text = std::move(g_lines.segs[pid.first].tr_tail);
		line.tr_text += iprefix;
		line.tr_text += line.name;
		line.tr_text += line.text;
		line.tr_text += '\n';
		line.tr_text += isuffix;
		line.tr_text += llama_out;
	}

	if (!is_stopped() && std::this_thread::get_id() == s_main_tid && g_mode == op_mode::rt_llama && id != c_bad_id) {
		// Launch worker thread to continue translation in the background
		// Check if the next line is untranslated
		if (auto next = g_lines.next(id); next != c_bad_id) {
			std::lock_guard lock(g_mutex);
			if (!g_lines[next].tr_text.empty())
				return true;
			g_lines[next].seed = 0;
			g_lines[next].tr_tts.clear();
		} else {
			// Nothing to do
			return true;
		}

		if (!g_lines.segs[id.first].tr_tail.empty())
			return true;

		// Make some space
		if (!eject_start())
			return false;

		stop_sig = -1;
		work_res = id.second + 1;
		worker = std::thread([=, &params] {
			auto start = g_lines.next(id);
			auto next = start;
			if (params.verbosity)
				std::fprintf(stderr, "%s[id:%u:%u] Thread entry\n", g_esc.reset, next.first, next.second);
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
						std::fprintf(stderr, "%s[id:%u:%u, from:%u] Cleaning\n", g_esc.reset, start.first, start.second, from);
					g_lines[start].tr_text.clear();
					g_lines[start].tr_tts.clear();
					eject_bunch(1); // reverse order but should work ok?
				}
				g_lines.advance(start);
			}
			if (params.verbosity)
				std::fprintf(stderr, "%s[id:%u:?] Thread exit\n", g_esc.reset, next.first);
		});
	}

	return true;
}
