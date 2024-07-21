# VNSleuth
This is an experimental tool for offline machine translation of Japanese Visual Novels to English. For translation, it's currently designed to use locally-running [LLMs](https://en.wikipedia.org/wiki/Large_language_model), very complex, resource-hungry neural networks. Translated text is displayed in a terminal in realtime, and original text needs to be hooked and copied to the clipboard by 3rdparty tools like [Textractor](https://github.com/Artikash/Textractor). There are no plans for packaging translated text back into a VN. The most interesting and promising thing about LLMs is that they are potentially able to generate coherent translation, while making sense of the previous sentences and overall setting, albeit not perfectly and not cheaply. And to do it properly, we also need to extract all the original text from the game files, which introduces more complications: different engines must be specifically supported, and extracting some of them might be tricky.

An example of running a VN with VNSleuth in the terminal on the right side. You can see Textractor icon on the left which is also running in the background, sending text to the clipboard.
![Screenshot of the 'Nekotsuku Sakura' with VNSleuth displaying some translation in the terminal.](/examples/screenshot1.png)

I made VNSleuth for myself out of curiosity for new tech and also laziness. Despite having started learning Japanese around 2011, I still don't know many ideograms. I used to read VNs with a dictionary, but it's time-consuming and I seemingly hit the wall. It seems I can achieve much better understanding by reading both Japanese and translated lines, and with decent speed and lower effort. I still don't want to use any sort of online translator however, unless on specific occasion.

## Requirements
1. VNSleuth is developed primarily on Linux, so Linux machine capable of running VNs, with Wine and other prerequisites. Xorg is required, Wayland support may be added later. I currently don't develop on Windows, but it might work with mingw. WSL might also work in theory. GUI might be added later, but I don't see much need for it.
1. Modern GPU with at least 16 GiB of VRAM for translating texts at acceptable speed. 4060 Ti or RX 7600 XT might work although the latter is probably the slowest option. It's possible to run partially (lower VRAM req.) or completely on CPU, but it's way too slow and power-hungry.
1. Technically you can use any model that's supported by [llama.cpp](https://github.com/ggerganov/llama.cpp), I'm currently testing [Gemma 27B](https://huggingface.co/bartowski/Big-Tiger-Gemma-27B-v1-GGUF/tree/main) — you can download any GGUF file that fits in your GPU memory. I have 24 GiB in total and use IQ4_XS for my tests — note that the model is much smaller than 24, this is because the memory usage will be much higher and you need some spare memory for system stability. MODEL_PATH refers to the .gguf file of the model you download.
1. Make sure `xsel` utility is installed on your system, via `sudo apt install xsel` on Ubuntu, for example. Other requirements include `git`, C++ compiler (g++), CMake.
1. Build `vnsleuth` with CMake and proper support for your GPU as documented in llama.cpp repository. Example for RX 7600 XT (**warning**: other GPUs should use different parameters)
```bash
git clone --recursive https://github.com/Nekotekina/VNSleuth
cd VNSleuth
mkdir build
# Build with support for gfx1102 architecture (just an example)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1102 -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -- -j $(nproc)
# Install an executable symlink to ~/bin (completely optional)
ln -sf "$(realpath ./build/vnsleuth)" "${HOME}/bin/vnsleuth"
# Build xclipmonitor accompanying tool (install is optional, requires XOrg and g++)
make -C tools/xclipmonitor install
```

## Workflow
1. Determine whether your VN's engine is supported. Extract all script files into a dedicated directory. Automated tool for script extraction may be added in future.

| Engine | Script location | Notes |
|:-------:|:-------:|:-------:|
| Buriko/ETH | data01000.arc | Extract data01000.arc with AE or GARbro |

2. Assuming you extracted scripts into `./Scripts/`, run `export P="./Scripts/" && vnsleuth "$P"` to initialize the script directory. You should see the number of script lines, dumped furigana and other technical information.
2. Make sure `./Scripts/__vnsleuth_names.txt` was created and filled with all encountered names. Optionally, you can fill some of them manually. This is not required as VNSleuth will attempt to translate them automatically and fill this file accordingly, but that process may fail sometimes due to the nature of LLMs. Add English names after `:`, without spaces, and make sure the line ends with another `:`.
2. Create `./Scripts/__vnsleuth_prompt.txt` that contains basic information about the VN (like protagonist name, heroines, toponyms). Consult /examples/ in the repository. There is no strict format to the prompt, but try to keep it brief and avoid repeating the same words. The most important part is probably the first line and "Dictionary". Character names are usually provided on https://vndb.org/ and dumped furigana can sometimes provide important word readings.
2. Run `xclipmonitor | vnsleuth "$P" -m MODEL_PATH`, other arguments like temperature may be specified too if you know what you are doing. Now you can run VN with Textractor (use the extension to send hooked text to the clipboard) or some other tool which can copy text to the clipboard automatically. Some hooks may work incorrectly, you may need to add appropriate custom hook if necessary. Press `Enter` in the terminal or use 5th mouse button ("Forward") to regenerate the translation for the latest sentence. If translation is bad no matter how many times you regenerate it, you can edit it manually. Exit vnsleuth first, edit translation cache file(s), then continue. You can easily find the newest cache file by sorting by date. It's beneficial to fix mistakes early because they can affect following automatic translation positively.
2. Alternatively, run `vnsleuth "$P" -m MODEL_PATH`. Wait for, possibly, **hours**, until the translation is finished. Not recommended anymore because you lose control over the translation. Then you can run `xclipmonitor | vnsleuth "$P"` (make sure there are no warnings). If full translation is cached, GPU support is not necessary.
