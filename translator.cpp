#include "common.h"
#include "llama.h"
#include "main.hpp"
#include "sampling.h"
#include "tools/tiny_sha1.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <emmintrin.h>
#include <immintrin.h>

static_assert("か"sv == "\xE3\x81\x8B"sv, "This source file shall be compiled as UTF-8 text");

extern volatile bool g_stop;

extern std::string g_neg;
extern std::string example;
extern std::string iprefix;
extern std::string isuffix;

extern std::string cache_prefix;

extern std::string print_line(line_id id, std::string* line, bool stream);

extern void update_segment(uint seg, bool upd_names, uint count);

namespace fs = std::filesystem;

//__attribute__((target("avx2")))
double cosine_similarity(const std::vector<float>& id1, double sum1, const std::vector<float>& id2, double sum2)
{
	double sum = 0.0;
	std::size_t len = id1.size();
	std::size_t i = 0;
#ifdef __SSE2__
	__m128d acc0 = _mm_setzero_pd();
	__m128d acc1 = _mm_setzero_pd();
	for (auto n = len & -4; i < n; i += 4) {
		auto v1 = _mm_load_ps(id1.data() + i);
		auto v2 = _mm_load_ps(id2.data() + i);
		auto v1l = _mm_cvtps_pd(v1);
		auto v1h = _mm_cvtps_pd(_mm_movehl_ps(v1, v1));
		auto v2l = _mm_cvtps_pd(v2);
		auto v2h = _mm_cvtps_pd(_mm_movehl_ps(v2, v2));
		acc0 = _mm_add_pd(acc0, _mm_mul_pd(v1l, v2l));
		acc1 = _mm_add_pd(acc1, _mm_mul_pd(v1h, v2h));
	}
	double sum_pair[2];
	_mm_storeu_pd(sum_pair, _mm_add_pd(acc0, acc1));
	sum = sum_pair[0] + sum_pair[1];
#endif
	for (auto n = id1.size(); i < n; i++) {
		sum += static_cast<double>(id1[i]) * id2[i];
	}

	// Handle the case where one or both vectors are zero vectors
	if (sum1 == 0.0 || sum2 == 0.0) {
		if (sum1 == 0.0 && sum2 == 0.0) {
			return 1.0; // two zero vectors are similar
		}
		return 0.0;
	}

	return sum / (sqrt(sum1) * sqrt(sum2));
}

decltype(g_replaces) make_name_replaces(std::string_view text)
{
	static constexpr std::pair<std::array<const char*, 4>, std::array<const char*, 4>> s_suffix_map[]{
		// clang-format off
		{{"さん"}, {"-san"}},
		{{"君", "くん"}, {"-kun"}},
		{{"様", "さま", "しゃま", "ちゃま"}, {"-sama", "-shama", "-chama"}},
		{{"ちゃん", "たん"}, {"-chan", "-tan"}},
		{{"どの", "殿"}, {"-dono"}},
		{{"先輩", "せんぱい", "センパイ"}, {"-senpai", "-sempai"}},
		{{"先生", "せんせい", "センセイ"}, {"-sensei"}}
		// clang-format on
	};

	decltype(g_replaces) result;

	for (const auto& [orig_name, pair] : g_dict) {
		// Filter non-names (TODO: this should be much more complicated)
		const auto& [tr_name, tr_ann] = pair;
		if (tr_name.empty() || tr_name.find_first_of("?& ") + 1)
			continue;
		std::string_view name = orig_name;
		name -= ":";
		std::size_t fpos = std::size(s_suffix_map), pos = 0;
		std::vector<char> features(fpos + 1, 0);
		while (pos < text.size()) {
			// TODO: convert tr_name to hiragana/katakana and use for search as well
			pos = text.find(name, pos);
			if (pos + 1 == 0)
				break;
			pos += name.size();
			bool found = false;
			// Test each suffix in set
			for (std::size_t f = 0; f < std::size(s_suffix_map); f++) {
				for (const auto& suf : s_suffix_map[f].first) {
					if (suf && text.substr(pos).starts_with(suf)) {
						found = true;
						features[f] |= 1;
						fpos = f;
						break;
					}
				}
			}
			if (!found)
				features.back() |= 1;
		}

		// Conflicts or nothing found
		if (std::accumulate(features.begin(), features.end(), 0) != 1)
			continue;
		std::string from(tr_name - ":");
		std::string to(from);
		if (!features.back()) {
			to += s_suffix_map[fpos].second[0];
			result.emplace_back(from, to);
		} else {
			// Disable "remove all suffixes" for now
			// Add dummy replace to indicate the presence of the name
			result.emplace_back(from, from);
			continue;
		}
		for (std::size_t f = 0; f < std::size(s_suffix_map); f++) {
			if (!features[f]) {
				for (const auto& tr_sfx : s_suffix_map[f].second) {
					if (tr_sfx)
						result.emplace_back(from + tr_sfx, to);
				}
			}
		}
	}

	return result;
}

// Return true if some meaningful CJK character is found (TODO: potentially incomplete)
bool check_cjk_line(std::u16string_view sq_text)
{
	for (char16_t c : sq_text) {
		// clang-format off
		if ((c >= '0' && c <= '9') || // ASCII numbers and letters are included too
			(c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= 0x3040 && c <= 0x30ff) || // Hiragana, katakana
			(c >= 0x4e00 && c <= 0x9fff) || // CJK ideograms
			(c >= 0xf900 && c <= 0xfaff) ||
			(c >= 0xac00 && c <= 0xd7af) || // Hangul precomposed
			(c >= 0x1100 && c <= 0x11ff) || // Hangul Jamo
			(c >= 0x3130 && c <= 0x318f) ||
			(c >= U'０' && c <= U'９') ||
			(c >= U'Ａ' && c <= U'Ｚ') ||
			(c >= U'ａ' && c <= U'ｚ') ||
			(c >= 0xff65 && c <= 0xff9f) || // Halfwidth kana
			c == '-')
			return true;
		// clang-format on
	}
	return false;
}

// Workaround for llama.cpp tokenize which expects correct utf-8
bool is_valid_utf8(std::string_view str)
{
	uint cont = 0;
	for (const char& c : str) {
		if ((c & 0x80) == 0) {
			// ASCII
			if (cont)
				return false;
		} else if ((c & 0xc0) == 0x80) {
			if (!cont)
				return false;
			cont--;
		} else if (cont) {
			return false;
		} else if ((c & 0xe0) == 0xc0) {
			cont = 1;
		} else if ((c & 0xf0) == 0xe0) {
			cont = 2;
		} else if ((c & 0xf8) == 0xf0) {
			cont = 3;
		} else {
			return false;
		}
	}
	return !cont;
}

std::vector<std::pair<uint, float>> get_recollections(common_params& params, line_id id, std::size_t pos_max)
{
	if (pos_max >= g_history.size()) {
		// Handle underflow from subtraction
		return {};
	}

	if (params.speculative.model == "." || !check_cjk_line(g_lines[id].sq_text)) {
		// Don't process lines like "..."
		return {};
	}

	// Vector of history positions
	std::vector<std::pair<uint, float>> result;
	auto& line = g_lines[id];

	// Relevancy mapping
	struct rel_ref {
		std::size_t pos;
		double sim;
	};
	std::deque<rel_ref> rel_map;

	for (std::size_t i = 0; i <= pos_max; i++) {
		// TODO: filter repetitions in the cause of history loops
		if (!check_cjk_line(g_lines[g_history[i]].sq_text))
			continue;
		auto& hline = g_lines[g_history[i]];
		// Exclude too big lines (TODO)
		if (hline.tokens > std::min(255u, params.n_ctx / 8u / 4u))
			continue;
		// More recent history appears first
		auto& rel = rel_map.emplace_front();
		rel.pos = i;
		rel.sim = cosine_similarity(line.embd, line.embd_sqrsum, hline.embd, hline.embd_sqrsum);
	}

	std::stable_sort(rel_map.begin(), rel_map.end(), [](const rel_ref& a, const rel_ref& b) {
		// Sort by similarity in descending order
		return a.sim > b.sim;
	});

	auto end = std::unique(rel_map.begin(), rel_map.end(), [](const rel_ref& a, const rel_ref& b) {
		// Filter duplicates (TODO: is it good idea?)
		// There is also a strange issue when equal strings get slightly different embeddings.
		auto& linea = g_lines[g_history[a.pos]];
		auto& lineb = g_lines[g_history[b.pos]];
		return &linea == &lineb || (linea.name == lineb.name && linea.text == lineb.text);
	});
	rel_map.erase(end, rel_map.end());

	uint tokens = 0;
	for (auto& rel : rel_map) {
		// Cut elements with low similarity
		auto& [pos, sim] = rel;
		tokens += g_lines[g_history[pos]].tokens;
		if (tokens > std::min(1024u, params.n_ctx / 8u) - 1) {
			break;
		}
		result.emplace_back(pos, sim);
	}

	// Sort result by history order
	std::sort(result.begin(), result.end());
	return result;
}

std::vector<llama_token> llama_tokenize(llama_model* model, std::string_view str, bool add_special, bool parse_special = false)
{
	std::vector<llama_token> result;
	result.resize(llama_n_ctx_train(model));
	auto count = llama_tokenize(model, str.data(), str.size(), result.data(), result.size(), add_special, parse_special);
	if (count < 0) {
		result.resize(-count);
		if (-count != llama_tokenize(model, str.data(), str.size(), result.data(), result.size(), add_special, parse_special))
			throw std::runtime_error("llama_tokenize");
	} else {
		result.resize(count);
	}
	return result;
}

std::string llama_token_to_piece(llama_model* model, llama_token t, bool special = true)
{
	std::string result;
	result.resize(1024);
	result.resize(llama_token_to_piece(model, t, result.data(), result.size(), 0, special));
	return result;
}

bool translate(common_params& params, line_id id, tr_cmd cmd)
{
	static const auto s_main_tid = std::this_thread::get_id();
	static std::atomic<uint> stop_sig = -1; // stop signal for thread, id.second to start discarding from
	static std::atomic<uint> work_res = 0;	// number of translated lines in segment, done by worker
	static std::condition_variable work_cv;
	static std::thread worker;

	static auto is_stopped = [](line_id id = c_bad_id) -> bool {
		if (g_stop)
			return true;
		if (std::this_thread::get_id() != s_main_tid) {
			if (uint sig = stop_sig.load(); sig + 1) {
				// Compare discard start pos with current id
				if (id.second >= sig)
					return true;
			}
		}
		return false;
	};

	static auto join_worker = [](uint val) {
		if (worker.joinable()) {
			std::lock_guard{g_mutex}, stop_sig = val;
			work_cv.notify_all();
			worker.join();
		}
	};

	// Initialize llama.cpp
	static const auto init_result = [&]() -> common_init_result {
		llama_backend_init();
		llama_numa_init(params.numa);
		return common_init_from_params(params);
	}();

	static const auto model = init_result.model;
	static const auto ctx = init_result.context;

	// Load embedding model if specified as "draft model"
	static const auto init_result_e = [&]() -> common_init_result {
		if (params.speculative.model == ".")
			return {nullptr, nullptr, {}};
		if (params.speculative.model.empty())
			return init_result;
		auto eparams = params;
		eparams.model = std::move(params.speculative.model);
		eparams.n_gpu_layers = params.speculative.n_gpu_layers;
		eparams.lora_adapters.clear();
		eparams.embedding = true;
		eparams.n_ctx = 0;		// auto
		eparams.n_ubatch = 512; //def
		eparams.n_batch = 2048; //def
		eparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
		eparams.flash_attn = true;
		eparams.cache_type_k = "f16";
		eparams.cache_type_v = "f16";
		return common_init_from_params(eparams);
	}();

	static const auto emodel = init_result_e.model;
	static const auto ectx = init_result_e.context;

	static std::vector<llama_token> tokens; // Current tokens (not necessarily decoded)
	static std::deque<uint> chunks;			// Size (in tokens) of each translated message block
	static uint decoded = 0;				// Number of token successfully llama_decode()'d
	static uint segment = -1;				// Current segment
	static uint hist_pos = -1;				// Position in g_history corresponding to chunks[0]
	static uint prompt_size = 0;			// Number of prompt tokens + reserved area for recollections
	static float penalty_scale = 1.f;

	static std::map<line_id, std::vector<std::uint8_t>> recollections;

	// Something like "$HOME/.cache/VNSleuth/GameXXX"
	static const fs::path s_cache_path = []() {
		fs::path result;
#if defined(__APPLE__)
		result = std::getenv("HOME");
		result /= "Library";
		result /= "Caches";
#elif defined(_WIN32)
		result = std::getenv("LOCALAPPDATA");
#else
		if (auto v = std::getenv("XDG_CACHE_HOME")) {
			result = v;
		} else {
			result = std::getenv("HOME");
			result /= ".cache";
		}
#endif
		result /= "VNSleuth";
		return result;
	}() / cache_prefix;

	static const struct _init_t {
		explicit operator bool() const { return model && ctx; }
		_init_t(common_params&)
		{
			if (!model || !ctx)
				return;
			if (llama_model_has_encoder(model))
				throw std::runtime_error("Decoder-only model expected");
			fs::create_directories(s_cache_path / "__embd");
		}
		~_init_t()
		{
			join_worker(0);
			if (ectx && ectx != ctx)
				llama_free(ectx);
			if (emodel && emodel != model)
				llama_free_model(emodel);
			llama_free(ctx);
			llama_free_model(model);
			llama_backend_free();
		}
	} _init{params};

	if (!_init) {
		std::cerr << "Failed to initialize llama model." << std::endl;
		return false;
	}

	if (cmd == tr_cmd::sync) {
		// Abort background worker and possibly discard work
		// Send Id to start discarding from, obviously can't use -1
		join_worker(std::min<uint>(id.second, -2));
		return true;
	}

	if (cmd == tr_cmd::kick && std::this_thread::get_id() == s_main_tid) {
		// Stop background worker first
		if (segment == id.first && worker.joinable()) {
			// Kick only once last translated line has been read
			uint tr_lines = work_res;
			uint rd_lines = id.second + 1;
			// Compare translated count with the number of read lines
			if (tr_lines > rd_lines) {
				// Only notify worker
				work_cv.notify_all();
				return true;
			} else if (tr_lines != rd_lines) {
				std::fprintf(stderr, "%sError: Kicked from untranslated line: %u<%u\n", g_esc.reset, tr_lines, rd_lines);
			}
		}

		join_worker(id.first != segment ? 0 : id.second + 1);
	}

	// Remove first message block (returns number of tokens to erase)
	static auto eject_first = [&params]() -> uint {
		if (chunks.empty()) {
			return 0;
		}

		auto count = chunks.front();
		chunks.pop_front();
		if (decoded > prompt_size) {
			// std::cerr << "*Used cells: " << llama_get_kv_cache_used_cells(ctx) << std::endl;
			// std::cerr << "*Decoded: " << decoded << std::endl;
			// std::cerr << "*Tokens: " << tokens.size() << std::endl;
			const auto p0 = prompt_size;
			const auto p1 = prompt_size + count;
			if (!llama_kv_cache_seq_rm(ctx, 0, p0, p1))
				throw std::runtime_error("llama_kv_cache_seq_rm 1");
			llama_kv_cache_seq_add(ctx, 0, p1, -1, 0u - count);
			decoded -= count;
			if (llama_get_kv_cache_used_cells(ctx) + 0u != decoded) {
				std::cerr << "Used cells: " << llama_get_kv_cache_used_cells(ctx) << std::endl;
				std::cerr << "Decoded: " << decoded << std::endl;
				// llama_kv_cache_view view = llama_kv_cache_view_init(ctx, 2);
				// llama_kv_cache_view_update(ctx, &view);
				// llama_kv_cache_dump_view(view);
				throw std::runtime_error("used cells overflow");
			}
			if (decoded > 0u + params.n_ctx)
				throw std::out_of_range("ctx underflow");
			if (decoded < prompt_size)
				throw std::out_of_range("eject_first failed");
		}
		hist_pos++;
		return count;
	};

	static const auto unload = []() {
		uint count = 0;
		while (!chunks.empty()) {
			count += eject_first();
		}
		tokens.clear();
		recollections.clear();
		decoded = 0;
		llama_kv_cache_clear(ctx);
		hist_pos = 0;
	};

	// Remove last message block(s)
	static auto eject_bunch = [&params](uint i, [[maybe_unused]] bool locked = true) {
		if (i > chunks.size()) {
			unload();
			if (llama_add_bos_token(model)) {
				tokens.push_back(llama_token_bos(model));
			}
			std::string buf;
			buf.resize(256);
			buf.resize(llama_model_desc(model, buf.data(), buf.size()));
			if (buf.starts_with("command-r ")) {
				penalty_scale = 3.f;
			}

			tokens += llama_tokenize(model, params.prompt + example, false);
			params.n_keep = tokens.size();
			prompt_size = tokens.size();
			if (ectx) {
				// Make space for recollections (currently constant)
				prompt_size += std::min(1024u, params.n_ctx / 8u);
				tokens.resize(prompt_size, llama_token_nl(model));
			}
			std::cerr << "Permanent tokens: " << prompt_size << std::endl;
			return;
		}
		while (i--) {
			auto count = chunks.back();
			chunks.pop_back();
			if (decoded == tokens.size()) {
				if (!llama_kv_cache_seq_rm(ctx, 0, decoded - count, -1))
					throw std::runtime_error("llama_kv_cache_seq_rm last");
				if (llama_get_kv_cache_used_cells(ctx) + 0u != tokens.size() - count)
					throw std::runtime_error("used cells in eject_bunch");
				decoded -= count;
				if (decoded > tokens.size() || decoded < prompt_size)
					throw std::out_of_range("eject_bunch decoded=" + std::to_string(decoded));
			}
			tokens.resize(tokens.size() - count);
			g_stats->raw_discards++;
		}
	};

	// Eject old translations if necessary
	static auto eject_start = [&params](bool defrag = false) -> bool {
		uint count = 0;
		while (tokens.size() - count + params.n_predict * 2 > params.n_ctx - 1u) {
			if (chunks.empty()) {
				std::cerr << "Prompt too big or context is too small" << std::endl;
				return false;
			}
			count += eject_first();
		}

		tokens.erase(tokens.begin() + prompt_size, tokens.begin() + (prompt_size + count));
		// Apply defrag
		if (defrag) {
			llama_kv_cache_defrag(ctx);
			llama_kv_cache_update(ctx);
		}
		return true;
	};

	// Tokenize line and add to the tokens
	static auto push_str = [&params](const std::string& text, bool front = false, bool spec = false) -> uint {
		auto tt = llama_tokenize(model, text, false, spec);
		if (params.verbosity && !tt.empty())
			std::fprintf(stderr, "%s[tokens:%zu,%d] push %zu tokens: '%s'\n", g_esc.reset, tokens.size(), +front, tt.size(), text.c_str());
		if (front) {
			tokens.insert(tokens.begin() + prompt_size + chunks.front(), tt.begin(), tt.end());
			chunks.front() += tt.size();
		} else {
			tokens.insert(tokens.end(), tt.begin(), tt.end());
			chunks.back() += tt.size();
		}
		return tt.size();
	};

	// Tokenize tr_text from id
	static auto push_id = [&params](line_id id, bool front = false) {
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
		auto out = apply_replaces(g_lines[id].text, false, 0);
		// Pre-annotations: no replaces, then original line with replaces
		g_lines[id].pre_ann = tr_text.substr(0, pref_pos);
		out = g_lines[id].pre_ann + iprefix + g_lines[id].name + std::move(out) + "\n";
		// Post-annotations: no replaces
		post_pos += iprefix.size();
		post_pos += g_lines[id].name.size();
		post_pos += g_lines[id].text.size() + 1;
		g_lines[id].post_ann = tr_text.substr(post_pos, spos - post_pos);
		out += g_lines[id].post_ann;
		out += isuffix;
		push_str(out, front);
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
			token_count += push_str(std::string(tr_text.substr(0, pos)), front);
			tr_text.remove_prefix(pos);
		}
		token_count += push_str(std::string(tr_text), front);
		g_lines[id].tr_tts.clear();
		g_lines[id].tr_tts.emplace_back(tokens.end() - token_count, tokens.end());
		if (token_count > params.n_predict + 1)
			throw std::runtime_error("Line too long: " + g_lines[id].tr_text);
		g_lines[id].tokens = front ? chunks.front() : chunks.back();
	};

	static auto decode_internal = [&params](uint count) -> void {
		if (params.verbosity) {
			std::cerr << "Decoding:" << g_esc.buf;
			auto end_it = tokens.begin() + decoded;
			if (tokens.end() - end_it > 300)
				end_it += 300;
			else
				end_it = tokens.end();
			for (auto it = tokens.begin() + decoded; it != end_it; it++) {
				auto str = llama_token_to_piece(model, *it, false);
				REPLACE(str, "\n", "\\n");
				std::cerr << str;
			}
			std::cerr << g_esc.reset << std::endl;
		}
		auto stamp0 = std::chrono::steady_clock::now();
		uint total = 0;
		if (count >= 0u + params.n_ctx)
			throw std::runtime_error("decode(): too many tokens: " + std::to_string(count));
		while (uint bsize = std::min<uint>(count, params.n_batch)) {
			// TODO: cannot properly interrupt by is_stopped, but probably not relevant anymore
			auto res = llama_decode(ctx, llama_batch_get_one(&tokens[decoded], bsize));
			if (res == 1) {
				llama_kv_cache_defrag(ctx);
				continue;
			}
			if (res < 0) {
				throw std::runtime_error("decode failed");
			}
			decoded += bsize;
			total += bsize;
			count -= bsize;
		}
		llama_synchronize(ctx);
		g_stats->raw_decodes += total;
		auto stamp1 = std::chrono::steady_clock::now();
		if (total > 1) {
			g_stats->batch_count += total;
			g_stats->batch_time += (stamp1 - stamp0).count() / 1000;
		}
	};

	static auto decode = [&params](line_id id, uint injected = 0) -> void {
		if (tokens.size() == decoded)
			return;
		if (decoded == 0) {
			// Decode prompt first
			decode_internal(prompt_size);
		} else if (injected) {
			if (id == c_bad_id) {
				std::fill(tokens.begin() + params.n_keep, tokens.begin() + prompt_size, llama_token_nl(model));
			}
			llama_kv_cache_seq_rm(ctx, 0, params.n_keep, prompt_size);
			auto batch = llama_batch_init(injected, 0, 1);
			for (uint i = 0; i < injected; i++) {
				common_batch_add(batch, tokens[params.n_keep + i], params.n_keep + i, {0}, false);
			}
			if (llama_decode(ctx, batch) != 0)
				throw std::runtime_error("llama_decode failed (injected)");
			llama_batch_free(batch);
			llama_synchronize(ctx);
			llama_kv_cache_seq_cp(ctx, 0, 1, params.n_keep, prompt_size);
			if (id != c_bad_id) {
				// Limit recollections cache
				while (recollections.size() >= params.speculative.n_min * 2u) {
					if (recollections.begin()->first > id)
						recollections.erase(std::prev(recollections.end()));
					else
						recollections.erase(recollections.begin());
				}
				// Cache recollections
				std::vector<std::uint8_t> buf;
				buf.resize(llama_state_seq_get_size(ctx, 1));
				buf.resize(llama_state_seq_get_data(ctx, buf.data(), buf.size(), 1));
				recollections[id] = std::move(buf);
			}
			llama_kv_cache_seq_keep(ctx, 0);
		}
		decode_internal(tokens.size() - decoded);
		if (llama_get_kv_cache_used_cells(ctx) + 0u != decoded) {
			throw std::runtime_error("used cells after decode");
		}
	};

	static auto inject_recollections = [&params](line_id id) -> uint {
		if (!ectx)
			return 0;
		const auto found = recollections.find(id);
		if (found != recollections.end()) {
			llama_kv_cache_seq_rm(ctx, 0, params.n_keep, prompt_size);
			llama_state_seq_set_data(ctx, found->second.data(), found->second.size(), 1);
			llama_kv_cache_seq_cp(ctx, 1, 0, params.n_keep, prompt_size);
			llama_kv_cache_seq_keep(ctx, 0);
			return 0;
		}
		std::lock_guard lock(g_mutex);
		uint injected = 0;
		std::fill(tokens.begin() + params.n_keep, tokens.begin() + prompt_size, llama_token_nl(model));
		if (auto inj_list = get_recollections(params, id, hist_pos - 1); !inj_list.empty()) {
			for (auto it = inj_list.begin(); it != inj_list.end(); it++) {
				auto iid = g_history[it->first];
				auto tts = llama_tokenize(model, g_lines[iid].tr_text, false);
				std::copy(tts.begin(), tts.end(), tokens.begin() + params.n_keep + injected);
				injected += tts.size();
			}
		}
		// Return constant number of tokens to decode
		return prompt_size - params.n_keep;
	};

	// Compose embedding cache filename from the hash of prompt+text
	static auto get_embd_file = [](line_id id) -> std::string {
		std::string result;
		char buf[42]{};
		auto& line = g_lines[id];
		{
			sha1::SHA1 s;
			int info[] = {llama_n_embd(emodel), llama_n_head(emodel), llama_n_layer(emodel), llama_n_vocab(emodel)};
			s.processBytes(&info, sizeof(info));
			//s.processBytes(line.name.data(), line.name.size());
			s.processBytes(line.text.data(), line.text.size());
			std::uint32_t digest[5];
			s.getDigest(digest);
			std::snprintf(buf, 41, "%08x%08x%08x%08x%08x", digest[0], digest[1], digest[2], digest[3], digest[4]);
		}
		result += buf;
		result += ".embf32";
		return s_cache_path / "__embd" / result;
	};

	static auto make_embedding = [](line_id id) -> void {
		auto esize = llama_n_embd(emodel);
		uint tsize = 0;
		std::vector<std::string> paths;
		std::vector<line_info*> lines;
		std::vector<std::vector<int>> tokens;
		// Try to batch as many embeddings as possible
		while (id != c_bad_id) {
			auto fname = get_embd_file(id);
			auto& line = g_lines[id];
			g_lines.advance(id);
			if (!line.embd.empty())
				continue;
			auto tt = llama_tokenize(emodel, "query: " + line.text, true);
			if (tt.size() > llama_n_ctx(ectx) - 1u)
				tt.resize(llama_n_ctx(ectx) - 1u);
			if (tsize + tt.size() > llama_n_ctx(ectx) - 1u)
				break;
			tsize += tt.size();
			paths.emplace_back(std::move(fname));
			lines.emplace_back(&line);
			tokens.emplace_back(std::move(tt));
			line.embd.resize(esize, 0.f);
		}
		if (paths.empty())
			return;
		llama_set_embeddings(ectx, true);
		llama_set_causal_attn(ectx, false);
		auto batch = llama_batch_init(llama_n_ctx(ectx), 0, 1);
		for (uint i = 0; i < lines.size(); i++) {
			for (uint j = 0; j < tokens[i].size(); j++) {
				common_batch_add(batch, tokens[i][j], j, {llama_seq_id(i + 1)}, true);
			}
		}
		if (llama_decode(ectx, batch) != 0)
			throw std::runtime_error("llama_decode failed (embd)");
		for (uint k = 0; k < tsize; k++) {
			// Use simple sum of each token's embeddings (TODO)
			auto* embd = llama_get_embeddings_ith(ectx, k);
			auto* line = lines[batch.seq_id[k][0] - 1];
			for (float& x : line->embd)
				x += *embd++;
		}
		llama_batch_free(batch);
		llama_kv_cache_seq_keep(ectx, 0);
		llama_set_embeddings(ectx, false);
		llama_set_causal_attn(ectx, true);
		// Save embeddings
		for (uint i = 0; i < paths.size(); i++) {
			auto& line = *lines[i];
			std::ofstream file(paths[i] + "~", std::ios::binary | std::ios::trunc);
			if (!file.is_open()) {
				throw std::runtime_error("Failed to create " + paths[i] + "~");
			}
			file.write(reinterpret_cast<char*>(line.embd.data()), esize * sizeof(float));
			file.close();
			fs::rename(paths[i] + "~", paths[i]);
			// Precompute ||embd||
			line.embd_sqrsum = 0;
			for (int i = 0; i < esize; i++) {
				line.embd_sqrsum += double(line.embd[i]) * line.embd[i];
			}
		}
	};

	static auto load_embedding = [](line_id id) -> void {
		if (!ectx)
			return;
		uint esize = llama_n_embd(emodel);
		auto fname = get_embd_file(id);
		auto& line = g_lines[id];
		std::ifstream file(fname, std::ios::binary);
		if (!file.is_open()) {
			make_embedding(id);
		} else {
			line.embd.resize(esize);
			line.embd_sqrsum = 0;
			file.read(reinterpret_cast<char*>(line.embd.data()), esize * sizeof(float));
			if (file.tellg() * sizeof(char) != esize * sizeof(float))
				throw std::runtime_error("Truncated embd file " + fname);
			for (uint i = 0; i < esize; i++) {
				line.embd_sqrsum += double(line.embd[i]) * line.embd[i];
			}
		}
	};

	static auto init_segment = [&]() -> void {
		// Initialize segment: eject all first
		eject_bunch(-1);
		decode(c_bad_id);

		// Invalidate embeddings
		for (auto& line : g_lines) {
			line.embd.clear();
			line.embd_sqrsum = 0.;
		}

		if (std::getenv("VNSLEUTH_DUMP_SIMILARITY")) {
			g_history.clear();
			std::ofstream dump(s_cache_path / (ctx == ectx ? "hdumpz.txt" : "hdump.txt"), std::ios_base::trunc);
			dump << std::fixed;
			std::cerr << std::endl;
			for (auto& line : g_lines) {
				auto id = line.get_id();
				g_history.push_back(id);
				line.tokens = llama_n_ctx(ectx) / 32;
				load_embedding(id);
				auto recs = get_recollections(params, id, g_history.size() - 2);
				std::sort(recs.begin(), recs.end(), [](auto& a, auto& b) { return a.second > b.second; });
				dump << "Line: " << g_lines.segs[id.first].src_name << ":" << id.second << std::endl;
				for (auto& [pos, sim] : recs) {
					auto line0 = g_lines[g_history[pos]];
					dump << "\t[" << sim << "] " << line0.name << line0.text << std::endl;
				}
				dump << "\t[->] " << line.name << line.text << std::endl;
				std::cerr << "\r" << g_history.size() << std::flush;
			}
			std::cerr << std::endl;
			dump.close();
			std::exit(0);
		}

		// Load full history
		for (auto& id : g_history) {
			if (g_lines[id].tr_text.empty())
				break;
			chunks.emplace_back();
			push_id(id);
			// Check cache files
			load_embedding(id);
		}

		if (!eject_start())
			throw std::runtime_error("eject_start init_seg");
		decode(c_bad_id);
	};

	auto add_tail_finalize = [id]() -> void {
		if (segment + 1) {
			auto& tail = g_lines.segs[segment].tr_tail;
			if (id.second == 0 && !g_lines.segs[segment].lines.back().tr_text.empty()) {
				// Finalize segment if necessary (TODO: try make it atomic with next segment creation)
				if (tail.empty()) {
					tail = "\n";
					update_segment(segment, true, -1);
				}
			}
		}
	};

	if (cmd == tr_cmd::eject) {
		eject_bunch(id.second);
		return true;
	}

	if (cmd == tr_cmd::unload) {
		unload();
		return true;
	}

	if (cmd == tr_cmd::reload) {
		if (id == c_bad_id || !decoded || (decoded - prompt_size) < (params.n_ctx - prompt_size) / 3u) {
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
				load_embedding(nid);
				if (!eject_start())
					throw std::runtime_error("eject_start reload");
				decode(nid);
			}
		}
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
			if (g_lines[nid].tr_tts.empty())
				throw std::runtime_error("tr_tts not found: " + g_lines[nid].text);
			// tr_tts is not discarded here
		}
		eject_bunch(to_eject);
	} else {
		return true;
	}

	// Detect additional lines for translation
	bool echo = false;
	for (line_id pid{segment, 0}; cmd == tr_cmd::translate && pid <= id; g_lines.advance(pid)) {
		auto& line = g_lines[pid];
		if (!(std::lock_guard{g_mutex}, line.tr_text.empty()))
			continue;

		auto lag0 = std::chrono::steady_clock::now();
		std::string llama_out;
		llama_out += line.pre_ann;
		if (pid < id || line.tr_tts.empty())
			echo = true; // Sticky flag to display source line with annotations, otherwise this is rewrite request
		if (std::this_thread::get_id() == s_main_tid && echo)
			std::cout << g_esc.orig << line.pre_ann << std::flush;
		std::string spker = print_line(pid, &llama_out, std::this_thread::get_id() == s_main_tid && echo);
		if (std::this_thread::get_id() == s_main_tid && echo)
			std::cout << g_esc.orig << line.post_ann << std::flush;
		llama_out += line.post_ann;
		llama_out += isuffix;
		chunks.emplace_back();
		push_str(llama_out);
		// Encode speaker separately and count its tokens
		int pred_count = 0;
		// Process rewrite request
		if (std::this_thread::get_id() == s_main_tid && !g_neg.empty()) {
			// Copy previous translation's tokens preceding selection
			llama_out.clear();
			std::vector<std::string> tstrs;
			for (auto t : line.tr_tts.back()) {
				tstrs.emplace_back(llama_token_to_piece(model, t));
				llama_out += tstrs.back();
				pred_count++;
				if (auto negpos = llama_out.find(g_neg); negpos + 1) {
					while (llama_out.size() > negpos) {
						llama_out -= tstrs.back();
						tstrs.pop_back();
						pred_count--;
					}
					while (!isuffix.ends_with(" ") && llama_out.ends_with(" ")) {
						llama_out -= tstrs.back();
						tstrs.pop_back();
						pred_count--;
					}
					break;
				}
			}
			if (params.verbosity)
				std::fprintf(stderr, "%s[tokens:%zu] push %zu tokens (repeated)\n", g_esc.reset, tokens.size(), tstrs.size());
			tokens.insert(tokens.end(), line.tr_tts.back().begin(), line.tr_tts.back().begin() + tstrs.size());
			chunks.back() += pred_count;
		} else {
			pred_count += push_str(spker);
			llama_out = spker;
		}
		if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid) {
			std::cout << g_esc.tran << llama_out.c_str() + llama_out.starts_with(" ");
			std::cout << std::flush; // Stream to terminal
		}
		if (!eject_start())
			return false;

		if (line.embd.empty()) {
			load_embedding(pid);
		}

		if (params.verbosity)
			std::fprintf(stderr, "%s[id:%u:%u, t:%zu, last:%u, chk:%llu] Line: %s%s\n", g_esc.reset, pid.first, pid.second, tokens.size(),
						 chunks.size() ? chunks.back() : -1, std::accumulate(tokens.begin(), tokens.end(), 0ull), line.name.c_str(),
						 line.text.data());

		// Inject context temporarily, decode pending tokens
		const uint injected = inject_recollections(pid);
		decode(pid, injected);
		{
			auto lag1 = std::chrono::steady_clock::now();
			g_stats->lag_count++;
			g_stats->lag_time += (lag1 - lag0).count() / 1000;
		}

		auto sparams = params.sampling;
		if (!line.tr_tts.empty())
			sparams.seed += std::accumulate(line.tr_tts.back().begin(), line.tr_tts.back().end(), line.tr_tts.size());
		const auto sctx = common_sampler_init(model, sparams);
		const bool use_grammar = !line.name.empty();
		for (int i = 0; i < pred_count; i++)
			common_sampler_accept(sctx, *(tokens.rbegin() + pred_count - i), use_grammar);
		static const auto real_isuffix = "\n" + isuffix;
		const auto replaces = make_name_replaces(line.text);
		std::size_t last_suf = llama_out.size();
		while (!is_stopped(pid) && !llama_out.ends_with("\n")) {
			// Predict next token
			auto stamp0 = std::chrono::steady_clock::now();
			auto logits = llama_get_logits(ctx);
			for (auto& vec : line.tr_tts) {
				if (pred_count + 0u >= vec.size())
					continue;
				const auto token_str = llama_token_to_piece(model, vec[pred_count]);

				// Workaround for not penalizing certain patterns
				static constexpr std::string_view s_no_penalty[]{
					" \"",
					" '", //
					" (", //
					" “", //
					"”",  //
					")",
				};
				if ((!line.name.empty() && llama_out == spker) || pred_count == 0) {
					bool found = false;
					for (auto s : s_no_penalty) {
						if (llama_out != spker) {
							while (s.starts_with(" "))
								s.remove_prefix(1);
						}
						if (token_str.starts_with(s)) {
							found = true;
							break;
						}
					}
					if (found)
						break;
				}
				if (std::equal(tokens.end() - pred_count, tokens.end(), vec.begin())) {
					// Don't penalize pure space if it appeared here before
					auto tstr = llama_token_to_piece(model, vec[pred_count], false);
					if (!isuffix.ends_with(" ") && token_str == " ")
						continue;
					// Penalize specific token (except newline)
					if (tstr != "\n")
						logits[vec[pred_count]] -= penalty_scale;
					// Penalize pure space (experimentally observed to appear sometimes and break the line)
					static const auto s_tspace = llama_tokenize(model, " ", false);
					logits[s_tspace[0]] -= penalty_scale;
				}
			}
			auto token_id = common_sampler_sample(sctx, ctx, -1);
			g_stats->raw_samples++;
			auto stamp1 = std::chrono::steady_clock::now();
			g_stats->sample_time += (stamp1 - stamp0).count() / 1000;
			static const auto s_tnl = llama_tokenize(model, "\n", false);
			if (++pred_count > params.n_predict)
				token_id = s_tnl[0]; // Force newline if size exceeded
			if (llama_token_is_eog(model, token_id))
				token_id = s_tnl[0]; // Force newline on EOT/EOS
			auto token_str = llama_token_to_piece(model, token_id);
			if (token_str == "\n")
				token_id = s_tnl[0]; // Fix alternative newline (TODO: correct workaround)
			if (std::count(token_str.begin(), token_str.end(), '\n') > 0u + token_str.ends_with("\n")) {
				// Attempt to fix the line containing incorrect newlines
				token_str.resize(token_str.find_first_of('\n') + 1);
				auto tks = llama_tokenize(model, token_str, false);
				if (tks.size() == 1)
					token_id = tks[0];
				else {
					std::cerr << "Token sequence was forsibly converted to newline: " << token_str;
					token_id = s_tnl[0];
				}
			}
			tokens.push_back(token_id);

			common_sampler_accept(sctx, token_id, use_grammar);
			llama_out += token_str;
			int to_decode = 1;
			int to_penalize = -1;
			auto print_tokens = [&]() {
				if (!params.verbosity)
					return;
				std::fprintf(stderr, "%sT: ", g_esc.reset);
				for (auto it = tokens.end() - pred_count; it != tokens.end(); it++) {
					std::fprintf(stderr, "%s|", llama_token_get_text(model, *it));
				}
				std::fprintf(stderr, " (dec:%d,pen='%s')\n", to_decode, to_penalize >= 0 ? llama_token_get_text(model, to_penalize) : "");
			};

			// Try to parse translated name
			if (spker.empty() && !line.name.empty()) {
				std::string_view speaker = llama_out;
				if (auto pos = speaker.find_first_not_of(" ") + 1)
					speaker.remove_prefix(pos - 1);
				speaker = speaker.substr(0, speaker.find(":") + 1);
				if (!speaker.empty()) {
					std::lock_guard lock(g_mutex);
					spker = g_dict[line.name].first = speaker;
				}
			}
			if ((!spker.empty() || line.name.empty()) && is_valid_utf8(llama_out)) {
				// TODO: implement separate replace system whith takes original text into account
				auto fixed = llama_out;
				// Apply name suffix replaces as well
				for (auto& [from, to] : replaces) {
					if (from == to) [[unlikely]] {
						// Special case (name was found without any known suffix)
						if (llama_out.ends_with(from) && llama_out == fixed) {
							// Penalize tokens starting with "-"
							to_penalize = -2;
							break;
						}
					} else if (auto nf = fixed.find(from, last_suf); nf + 1) {
						// Completely cut the continuation
						last_suf = nf + to.size();
						fixed.resize(nf);
						fixed += to;
						break;
					}
				}
				// Retokenize to also make the tokenization uniform
				auto tt = llama_tokenize(model, fixed, false);
				if (!fixed.empty() && (fixed != llama_out || !std::equal(tt.begin(), tt.end(), tokens.end() - pred_count))) {
					if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid) {
						// Overwrite current line
						if (fixed.size() < llama_out.size()) {
							std::cout << std::string(llama_out.size() - fixed.size(), ' ');
							std::cout << std::string(llama_out.size() - fixed.size(), '\b');
						}
						std::cout << std::string(llama_out.size(), '\b') << g_esc.tran << fixed.c_str() + (fixed.starts_with(" "));
						std::cout << std::flush;
					}
					// Replace only the few last tokens
					auto [mis, mistt] = std::mismatch(tokens.end() - pred_count, tokens.end(), tt.begin(), tt.end());
					uint miscount = tokens.end() - mis;
					int dropped = mis != tokens.end();
					if (mistt == tt.end()) {
						// If nothing to replace, re-evaluate last token and penalize token that was following it
						// TODO: better logic
						if (llama_token_to_piece(model, *mis, false) != " ")
							to_penalize = *mis;
						to_decode++;
					}
					tokens.erase(mis, tokens.end());
					int to_remove = miscount - dropped;
					if (mistt == tt.end()) {
						to_remove++;
					}
					to_decode += tt.end() - mistt;
					to_decode -= dropped;
					tokens.insert(tokens.end(), mistt, tt.end());
					chunks.back() -= to_remove;
					const auto p0 = decoded - to_remove;
					decoded -= to_remove;
					if (!llama_kv_cache_seq_rm(ctx, 0, p0, -1))
						throw std::runtime_error("llama_kv_cache_seq_rm replaces");
					common_sampler_reset(sctx);
					for (auto& t : tt)
						common_sampler_accept(sctx, t, use_grammar);
					llama_out = std::move(fixed);
					pred_count += to_decode - to_remove - 1;
					token_str.clear();
				}
			}
			print_tokens();
			if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid && !token_str.empty()) {
				std::cout << token_str.c_str() + (token_str.starts_with(" ") && llama_out == token_str);
				std::cout << std::flush; // Stream to terminal
			}

			if (tokens.size() >= 0u + params.n_ctx)
				throw std::runtime_error("eval: too many tokens: " + std::to_string(tokens.size()));
			if (decoded >= tokens.size())
				throw std::runtime_error("eval: bad pointer: " + std::to_string(decoded));
			stamp0 = std::chrono::steady_clock::now();
			if (auto result = llama_decode(ctx, llama_batch_get_one(&tokens[decoded], to_decode))) {
				std::cerr << "llama_decode failed: " << result << std::endl;
				return false;
			}
			llama_synchronize(ctx);
			g_stats->raw_decodes += to_decode;
			stamp1 = std::chrono::steady_clock::now();
			g_stats->eval_time += (stamp1 - stamp0).count() / 1000;
			decoded += to_decode;
			if (chunks.empty())
				throw std::out_of_range("chunks.empty()");
			chunks.back() += to_decode;
			if (decoded > 0u + params.n_ctx)
				throw std::out_of_range("ctx overflow");
			logits = llama_get_logits(ctx);
			if (to_penalize >= 0) {
				logits[to_penalize] -= penalty_scale;
			} else if (to_penalize == -2) {
				for (int i = 0; i < llama_n_vocab(model); i++) {
					if (llama_token_get_text(model, i)[0] == '-') {
						logits[i] -= penalty_scale;
					}
				}
			}
		}

		common_sampler_free(sctx);
		g_stats->sample_count += pred_count;
		if (is_stopped(pid) && !llama_out.ends_with("\n") && std::this_thread::get_id() == s_main_tid)
			std::cout << std::endl; // Stop current line
		if (g_mode == op_mode::rt_llama && std::this_thread::get_id() == s_main_tid)
			std::cout << g_esc.reset << std::flush; // Reset terminal colors
		if (is_stopped(pid) && !llama_out.ends_with("\n")) {
			// Discard incomplete work
			eject_bunch(1, false);
			return true;
		}

		if (spker.empty() && !line.name.empty())
			std::cerr << "Failed to parse speaker translation for: " << line.name << std::endl;

		// Store translation tokens for possible reuse
		line.tr_tts.emplace_back(tokens.end() - pred_count, tokens.end());

		// Store translated line
		std::lock_guard lock(g_mutex);
		line.tr_text.clear();
		line.tr_text += line.pre_ann;
		line.tr_text += iprefix;
		line.tr_text += line.name;
		line.tr_text += line.text;
		line.tr_text += '\n';
		line.tr_text += line.post_ann;
		line.tr_text += isuffix;
		line.tr_text += llama_out;
		line.tokens = chunks.back();
		g_stats->raw_translates++;
	}

	if (cmd == tr_cmd::translate)
		return true;

	if (!is_stopped() && std::this_thread::get_id() == s_main_tid && g_mode == op_mode::rt_llama && id != c_bad_id) {
		// Launch worker thread to continue translation in the background
		// Check if thread is disabled:
		if (params.speculative.n_min <= 0)
			return true;
		// Check if the next line is untranslated
		if (auto next = g_lines.next(id); next != c_bad_id) {
			std::lock_guard lock(g_mutex);
			if (!g_lines[next].tr_text.empty())
				return true;
		} else {
			// Nothing to do
			return true;
		}

		stop_sig = -1;
		work_res = id.second + 1;
		worker = std::thread([=, &params] {
			std::deque<uint> old_pos;
			old_pos.push_back(hist_pos);
			auto start = g_lines.next(id);
			auto next = start;
			auto last = id;
			for (int i = 0; i < params.speculative.n_min; i++) {
				last.second++;
				if (g_lines.is_last(last))
					break;
			}
			uint ahead = 0;
			if (params.verbosity)
				std::fprintf(stderr, "%s[id:%u:%u] Thread entry (%d)\n", g_esc.reset, next.first, next.second, int(cmd));
			while (!is_stopped() && next != c_bad_id) {
				if (next <= last) {
					if (!translate(params, next, tr_cmd::translate)) {
						// No space left in context
						throw std::runtime_error("Unexpected: no space in context");
					}
					work_res++;
					g_lines.advance(next);
					ahead += 1;
					old_pos.push_back(hist_pos);
					continue;
				}
				std::unique_lock lock(g_mutex);
				while (!is_stopped() && next > last) {
					ahead = last.second - g_history.back().second;
					// Fix negative values, should mean untranslated lines were added to history
					for (auto it = g_history.rbegin(); it != g_history.rend(); it++) {
						if (!g_lines[*it].tr_text.empty())
							break;
						ahead++;
					}
					if (ahead > INT_MAX)
						throw std::runtime_error("Unexpected ahead: " + std::to_string(ahead));
					for (int i = 0; i < params.speculative.n_min; i++) {
						if (ahead + i < params.speculative.n_min + 0u) {
							last.second++;
							if (g_lines.is_last(last))
								break;
						} else
							break;
					}
					if (next <= last)
						break;
					work_cv.wait(lock);
				}
			}
			// Wait for thread's destiny instead of terminating it immediately
			std::unique_lock lock(g_mutex);
			while (stop_sig == 0u - 1)
				work_cv.wait(lock);
			// Discard translations which were done in the background
			bool msg = false;
			uint from = stop_sig.load();
			while (start != c_bad_id && !g_lines[start].tr_text.empty()) {
				if (start.second >= from) {
					if (params.verbosity && !std::exchange(msg, true))
						std::fprintf(stderr, "%s[id:%u:%u, from:%u] Cleaning\n", g_esc.reset, start.first, start.second, from);
					g_lines[start].tr_text.clear();
					g_lines[start].tr_tts.clear();
					eject_bunch(1); // reverse order but should work ok?
					old_pos.pop_back();
				}
				g_lines.advance(start);
			}
			if (llama_get_kv_cache_used_cells(ctx) + 0u != decoded) {
				throw std::runtime_error("used cells before thread exit");
			}
			if (params.verbosity)
				std::fprintf(stderr, "%s[id:%u:?] Thread exit\n", g_esc.reset, next.first);
		});
	}

	return true;
}
