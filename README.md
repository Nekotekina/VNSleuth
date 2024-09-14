# VNSleuth
This is an experimental tool for AI-assisted translation of (primarily) Japanese Visual Novels to English. Translated text is displayed in a terminal in realtime as VN progresses, and can be edited (fixed) in-place. Here is some video example of using it.

[![VNSleuth pre-0.4 example](https://img.youtube.com/vi/ae5iq6ImwtQ/0.jpg)](https://www.youtube.com/watch?v=ae5iq6ImwtQ)

Features:
- Written in C++ without external dependencies except llama.cpp (which is written in C++ too and only needs hardware-specific libraries like CUDA or HIP).
- Arbitrarily written translation prompt (contains brief description of characters and translation of specific terms or names).
- Context-aware translation using prompt information and previous lines. Prompt is always "visible" for the translator, but normal lines are gradually ejected.
- "Recollections": injection of very distant lines which seem relevant, to compensate for limited context window to further improve translation coherency.
- "Rewrite" (full or partial). If translation seems wrong or poorly written, you can "retry", optionally selecting the part of text to start rewriting from.
- Manual edit (implemented via launching external text editor) in case automatic translation fails.
- Free-form annotations (requires manual edit) for manually providing additional context-specific information. For example, change of the point of view or important unspoken effects.
- QoL features like automatic fixing of incorrectly translated name suffix (-san, -sama, etc) and custom replacement table (not really needed in most cases).
- It should be possible to change source and destination languages, but it's not tested and English translation is expected to have much better quality.

I made VNSleuth for myself out of curiosity for new tech and also laziness. Despite having started learning Japanese around 2011, I still don't know many ideograms. I used to read VNs with a dictionary, but it's time-consuming and I seemingly hit the wall. It seems I can achieve much better understanding by reading both Japanese and translated lines, and with decent speed and lower effort.

## Requirements
1. Original text needs to be hooked and copied to the clipboard by 3rdparty tools like [Textractor](https://github.com/Artikash/Textractor).
1. Not only that, but VNSleuth needs to support specific game engine to extract and preprocess its text (see the support table below).
1. VNSleuth is developed primarily on Linux, so Linux machine capable of running VNs, with Wine and other prerequisites. I currently don't develop on Windows, but it might work with mingw. WSL might also work in theory. GUI might be added later, but I don't see much need for it.
1. Modern GPU with at least 16 GiB of VRAM for translating texts at acceptable speed. 4060 Ti or RX 7600 XT might work although the latter is probably the slowest option. It's possible to run partially (lower VRAM req.) or completely on CPU, without any GPU at all, but it's way too slow and power-hungry.
1. For translation, it's currently designed to use locally-running [LLMs](https://en.wikipedia.org/wiki/Large_language_model), very massive, resource-hungry neural networks. There are many varieties of them freely available online on huggingface. Technically you can use any model that's supported by [llama.cpp](https://github.com/ggerganov/llama.cpp), I'm currently testing [Gemma-2 27B](https://huggingface.co/bartowski/gemma-2-27b-it-GGUF) — you can download any GGUF file that fits in your GPU memory (you only need one). I use mainly IQ4_XS quant for my tests — note that the model requires more VRAM. For 16 GiB VRAM, I recommend gemma-2-27b-it-IQ3_M.gguf. MODEL_PATH refers to the .gguf file of the model you download.
1. Additional model for embeddings (used internally for searching relevant lines), referred as E5_PATH. [multilingual-e5](https://huggingface.co/chris-code/multilingual-e5-large-Q8_0-GGUF).
1. SSD with significant amount of space. Usage depends on the model. ~500 G should be enough for one big VN. It will probably be made optional in future.
1. Make sure `xsel` utility is installed on your system, via `sudo apt install xsel` on Ubuntu, for example. `xclipmonitor` uses xsel internally and assumes Xorg. Other requirements include `git`, C++ compiler (g++), CMake.
1. Build `vnsleuth` with CMake and proper support for your GPU as documented in llama.cpp repository. Example for RX 7600 XT (**warning**: other GPUs should use different parameters)
```bash
git clone --recursive https://github.com/Nekotekina/VNSleuth
cd VNSleuth
mkdir build
# Build with support for gfx1102 architecture (just an example)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1102 -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -- -j $(nproc)
# Install an executable symlink to ~/bin (completely optional)
ln -sf "$(realpath ./build/vnsleuth)" "${HOME}/bin/vnsleuth"
# Install an optional tool to edit last file with gedit
ln -sf "$(realpath ./tools/lastfile)" "${HOME}/bin/lastfile"
# Build xclipmonitor accompanying tool (install is optional, requires XOrg and g++)
make -C tools/xclipmonitor install
```

## Workflow
1. Determine whether your VN's engine is supported.

| Engine | Script location | Notes |
|:-------:|:-------:|:-------:|
| Buriko/ETH | data01000.arc | Supported |

2. Assuming the game location `~/Games/Game/`, run `export P=~/Games/Game && vnsleuth "$P" --check` to initialize __vnsleuth directory. You should see the number of script lines, dumped furigana and other technical information.
2. Make sure `Game/__vnsleuth/__vnsleuth_names.txt` was created and filled with all encountered names. Optionally, you can fill some of them manually. This is not required as VNSleuth will attempt to translate them automatically and fill this file accordingly, but that process may fail sometimes due to the nature of LLMs. Add English names after `:`, without spaces, and make sure the line ends with another `:`.
2. Create `Game/__vnsleuth/__vnsleuth_prompt.txt` that contains basic information about the VN (like protagonist name, heroines, toponyms). Consult /examples/ in the repository. There is no strict format to the prompt, but try to keep it brief and avoid repeating the same words. The most important part is probably the first line and "Dictionary". Character names are usually provided on https://vndb.org/ and dumped furigana can sometimes provide important word readings.
2. Run `xclipmonitor lastfile "$P" | vnsleuth "$P" -m MODEL_PATH -md E5_PATH -co`, other arguments like temperature may be specified too if you know what are you doing. Now you can run VN with Textractor (use the extension to send hooked text to the clipboard) or some other tool which can copy text to the clipboard automatically. Some hooks may work incorrectly, you may need to add appropriate custom hook if necessary. Press `Enter` in the terminal or use 5th mouse button ("Forward") to regenerate the translation for the latest sentence. If translation is bad no matter how many times you regenerate it, you can edit it manually with command `e`. It's beneficial to fix mistakes early because they can affect following automatic translation positively. Other commands supported by xclipmonitor are `r` (reload) and `s` (save), but you probably don't need them.

## License
GPLv3 if not specified in specific file, some files have more permissive licenses.