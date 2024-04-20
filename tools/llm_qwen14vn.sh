#!/bin/bash
# -c 4096 is context size. More requires more VRAM.
# -n 128 is max size of the response.
# -l options: penalize ??? token (33015); " ---" token (12448); JP tokens (2 variants 27188 and 48780)
llm_qwen14.sh -l 33015-0 -l 12448-1 -l 27188-5 -l 48780-5\
	-c 4096\
	-e -r "\\n"\
	--interactive-first\
	--keep -1\
	-n 128\
	--temp 0.2\
	--top-p 0.3\
	--repeat-last-n 3\
	--repeat-penalty 1.1\
	--ignore-eos\
	--logit-restrict " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~-=\!\?.,:;'"\""%&()\n"\
	"$@"
