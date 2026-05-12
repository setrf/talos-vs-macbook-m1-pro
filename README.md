# microGPT pos=5 breakthrough

Tiny TALOS-V2/microGPT inference on Apple Silicon, reduced to the strict
pos=5 cumulative-table breakthrough.

## Result

Validated single-stream result:

```text
median: 19,570,123 tok/sec
best:   22,028,712 tok/sec
```

Correctness gates:

```text
./bench --check 10000000
strict check ok: 10000000 tokens

./bench --driver-check 10000000
driver check ok: 10000000 tokens
```

## Idea

The model has only 27 possible tokens. For the first six token positions
(`pos=0..5`), every possible history is small enough to enumerate exactly.

This benchmark precomputes exact cumulative next-token distributions for
all of those early histories. Runtime still samples normally from the same
RNG and produces the same token stream as the strict reference path.

This is not a generated-output cache. It is exact distribution
precomputation.

Tradeoff: the pos=5 table uses roughly 1.8 GiB of memory at startup.

## Run

```bash
make
make check
./bench 10000000 300000
```

The first build downloads `weights_only.npy` from TALOS-V2 and writes
`assets/weights_fp32.bin`.

## Files

```text
bench_c_strict.c   strict fp32 NEON benchmark
convert_weights.py weight downloader/converter
Makefile           build/check commands
LICENSE            upstream MIT license
```
