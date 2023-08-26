## baby-llama2.cpp

This is a minimal port of [llama2.c](https://github.com/karpathy/llama2.c) to C/C++

This repo is not self-contained, original [llama2.c](https://github.com/karpathy/llama2.c) files are needed, and then `run.cpp` and `Makefile`can be used as a direct drop-in replacement for the original ones.

Regarding coding style and flow, I try to stick to the original, there are few deviations where, under my very personal opinion, code can be improved.

Of course using C++ along with C gives some advantages: std library makes many little things easier and more straightforward.

Regarding functionality and performance, `run.cpp` should be 100% equivalent with upstream.
