CC ?= clang
CFLAGS := -O3 -mcpu=native -ffast-math -fno-stack-protector -fomit-frame-pointer -pthread -Wall -Wextra -Wno-unused-parameter
DEFS := -DSTRICT_POS1_LUT -DSTRICT_POS01_MLP_LUT -DSTRICT_POS01_LOGITS_LUT -DSTRICT_POS01_CUM_LUT -DSTRICT_UNROLLED_DRIVER -DSTRICT_QOS -DSTRICT_SAMPLE_NEON -DSTRICT_SAMPLE_NEON_U64MASK -DSTRICT_MLP_UNLIKELY -DSTRICT_MLP_ROW_HINTS -DSTRICT_QK_PAIR_LUT -DSTRICT_DRIVER_POS01_FAST -DSTRICT_DRIVER_POS2_FAST -DSTRICT_SOFTMAX_MAX_SWITCH -DSTRICT_MLP_W1_PREFETCH -DSTRICT_POS2_CUM_LUT -DSTRICT_POS3_CUM_LUT -DSTRICT_POS4_CUM_LUT -DSTRICT_POS5_CUM_LUT -DBENCH_LABEL='"c fp32+NEON strict-pos5-cum"'

all: bench

bench: bench_c_strict.c assets/weights_fp32.bin
	$(CC) $(CFLAGS) $(DEFS) $< -o $@

assets/weights_fp32.bin: convert_weights.py
	python3 convert_weights.py

check: bench
	./bench --check 10000000
	./bench --driver-check 10000000

clean:
	rm -f bench

.PHONY: all check clean
