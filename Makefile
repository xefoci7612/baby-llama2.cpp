# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = g++
CFLAGS = -Wall -Wextra -Wno-unknown-pragmas

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.cpp
	$(CC) $(CFLAGS) -O3 -o run run.cpp

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.cpp
	$(CC) $(CFLAGS) -g -O1 -o run run.cpp

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.cpp
	$(CC) $(CFLAGS) -Ofast -o run run.cpp

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.cpp
	$(CC) $(CFLAGS) -Ofast -fopenmp -march=native run.cpp -o run

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc-win32 $(CFLAGS) -Ofast -D_WIN32 -o run.exe -I. run.cpp win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) $(CFLAGS) -Ofast -std=gnu11 -o run run.cpp

.PHONY: runompgnu
runompgnu:
	$(CC) $(CFLAGS) -Ofast -fopenmp -std=gnu11 run.cpp -o run

.PHONY: clean
clean:
	rm -f run
