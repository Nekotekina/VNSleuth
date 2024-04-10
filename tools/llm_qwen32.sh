#!/bin/bash
# Fix downloaded gguf file location.
# -c 4096 is context size. More requires more VRAM.
# -ngl 999 is set to offload all layers to the GPU.
llm_common.sh -m ~/Downloads/qwen1.5-32b-chat-imat-IQ4_XS.gguf -c 4096 -ngl 999 "$@"
