#!/bin/bash
# Fix path to the main llama.cpp executable.
# -s 0 means setting random seed to fixed value, to produce repeatable output.
~/llama.cpp/build/bin/main -s 0 -n -1 --keep -1 --repeat-last-n 1024 "$@"
