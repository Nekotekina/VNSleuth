#!/bin/bash
# -c 4096 is context size. More requires more VRAM.
# -l options: penalize ??? token (33015); --- token; JP tokens (2 variants 27188 and 48780); EN token (5190)
llm_qwen32.sh -c 4096 -l 33015-0 -l 12448-1 -l 27188-5 -l 48780-5 -l 5190-5 \
	-e -r "\\n"\
	--logit-restrict " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~-=!?.,:;'"\""%&()" "$@"
