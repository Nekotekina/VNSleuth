#!/bin/bash
# Fix downloaded gguf file location.
# -c 4096 is context size. More requires more VRAM.
# -ngl 41 is set to offload all layers to the GPU.
llm_common.sh -m ~/Downloads/qwen1_5-14b-chat.Q5_K_M.gguf -c 4096 -ngl 41 "$@"
