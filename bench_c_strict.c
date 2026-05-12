#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <arm_neon.h>

#define VOCAB 27
#define BLOCK 16
#define EMBD 16
#define HEAD 4
#define HD 4
#define MLP_H 64
#define BOS 26
#define TEMP 0.5f
#define INV_TEMP (1.0f / TEMP)
#define HOT_INLINE static __attribute__((always_inline)) inline
#ifndef BENCH_LABEL
#define BENCH_LABEL "c fp32+NEON strict"
#endif
#ifndef STRICT_MLP_W1_PREFETCH_DIST
#define STRICT_MLP_W1_PREFETCH_DIST 8
#endif
#ifndef STRICT_MLP_W2T_PREFETCH_DIST
#define STRICT_MLP_W2T_PREFETCH_DIST STRICT_MLP_W1_PREFETCH_DIST
#endif
#ifndef STRICT_MLP_W1_PREFETCH_LOCALITY
#define STRICT_MLP_W1_PREFETCH_LOCALITY 3
#endif
#ifndef STRICT_MLP_W2T_PREFETCH_LOCALITY
#define STRICT_MLP_W2T_PREFETCH_LOCALITY STRICT_MLP_W1_PREFETCH_LOCALITY
#endif
#define LOGIT_STRIDE 32
#define LIVE_LOGIT_STRIDE 48
#define W2T_STRIDE 32
#define MLP_POSITIVE(h) __builtin_expect((h) > 0.0f, 0)
#define MLP_ROW_LIKELY(row_) ( \
    (row_) == 21 || (row_) == 27 || (row_) == 28 || (row_) == 34 || \
    (row_) == 50 || (row_) == 62)
#define MLP_POSITIVE_ROW(row_, h_) __builtin_expect((h_) > 0.0f, MLP_ROW_LIKELY(row_))
#define STEP_T_TYPE uint8_t
#define STEP_TP(T_, t_) TP_INDEX((T_)[(t_)], (t_))
#define STEP_T_PARAM , STEP_T_TYPE *restrict T
#define STEP_T_ARG , T
#define STEP_T_ARG_NAMED(name) , name
#define STEP_V_PTR(T_, V_, t_, hi_) ((V_) + (t_) * EMBD + (hi_) * HD)

typedef struct {
    float x[EMBD] __attribute__((aligned(64)));
    float logits[LIVE_LOGIT_STRIDE] __attribute__((aligned(64)));
    float attn[BLOCK] __attribute__((aligned(64)));
} StepScratch;


#define OFF_WTE 0
#define OFF_WPE (OFF_WTE + VOCAB * EMBD)
#define OFF_WQ  (OFF_WPE + BLOCK * EMBD)
#define OFF_WK  (OFF_WQ  + EMBD * EMBD)
#define OFF_WV  (OFF_WK  + EMBD * EMBD)
#define OFF_WO  (OFF_WV  + EMBD * EMBD)
#define OFF_W1  (OFF_WO  + EMBD * EMBD)
#define OFF_W2  (OFF_W1  + MLP_H * EMBD)
#define OFF_LM  (OFF_W2  + EMBD * MLP_H)
#define TOTAL   (OFF_LM  + VOCAB * EMBD)

static float W[TOTAL] __attribute__((aligned(64)));
static struct { char pad[64]; float v[VOCAB * BLOCK * EMBD] __attribute__((aligned(64))); } XR_LUT_STORAGE __attribute__((aligned(64)));
static struct { char pad[128]; float v[VOCAB * BLOCK * EMBD] __attribute__((aligned(64))); } Q_LUT_STORAGE __attribute__((aligned(64)));
static struct { char pad[192]; float v[VOCAB * BLOCK * EMBD] __attribute__((aligned(64))); } K_LUT_STORAGE __attribute__((aligned(64)));
static struct { char pad[256]; float v[VOCAB * BLOCK * EMBD] __attribute__((aligned(64))); } V_LUT_STORAGE __attribute__((aligned(64)));
static struct { char pad[320]; float v[VOCAB * BLOCK * HEAD] __attribute__((aligned(64))); } QK_SELF_LUT_STORAGE __attribute__((aligned(64)));
#define XR_LUT XR_LUT_STORAGE.v
#define Q_LUT Q_LUT_STORAGE.v
#define K_LUT K_LUT_STORAGE.v
#define V_LUT V_LUT_STORAGE.v
#define QK_SELF_LUT QK_SELF_LUT_STORAGE.v
static struct { char pad[384]; float v[VOCAB * BLOCK * VOCAB * BLOCK * HEAD] __attribute__((aligned(64))); } QK_PAIR_LUT_STORAGE __attribute__((aligned(64)));
#define QK_PAIR_LUT QK_PAIR_LUT_STORAGE.v
static struct { char pad[448]; float v[VOCAB * EMBD] __attribute__((aligned(64))); } X_POS0_LUT_STORAGE __attribute__((aligned(64)));
#define X_POS0_LUT X_POS0_LUT_STORAGE.v
static struct { char pad[512]; float v[VOCAB * EMBD] __attribute__((aligned(64))); } X_POS1_LUT_STORAGE __attribute__((aligned(64)));
#define X_POS1_LUT X_POS1_LUT_STORAGE.v
static struct { char pad[576]; float v[VOCAB * EMBD] __attribute__((aligned(64))); } X2_POS0_LUT_STORAGE __attribute__((aligned(64)));
#define X2_POS0_LUT X2_POS0_LUT_STORAGE.v
static struct { char pad[640]; float v[VOCAB * EMBD] __attribute__((aligned(64))); } X2_POS1_LUT_STORAGE __attribute__((aligned(64)));
#define X2_POS1_LUT X2_POS1_LUT_STORAGE.v
static struct { char pad[704]; float v[VOCAB * LOGIT_STRIDE] __attribute__((aligned(64))); } LOGITS_POS0_LUT_STORAGE __attribute__((aligned(64)));
#define LOGITS_POS0_LUT LOGITS_POS0_LUT_STORAGE.v
static float MAXL_POS0_LUT[VOCAB] __attribute__((aligned(64)));
static uint8_t MAXI_POS0_LUT[VOCAB] __attribute__((aligned(64)));
static struct { char pad[768]; float v[VOCAB * LOGIT_STRIDE] __attribute__((aligned(64))); } CUM_POS0_LUT_STORAGE __attribute__((aligned(64)));
#define CUM_POS0_LUT CUM_POS0_LUT_STORAGE.v
static float SUM_POS0_LUT[VOCAB] __attribute__((aligned(64)));
static struct { char pad[832]; float v[VOCAB * LOGIT_STRIDE] __attribute__((aligned(64))); } LOGITS_POS1_LUT_STORAGE __attribute__((aligned(64)));
#define LOGITS_POS1_LUT LOGITS_POS1_LUT_STORAGE.v
static float MAXL_POS1_LUT[VOCAB] __attribute__((aligned(64)));
static uint8_t MAXI_POS1_LUT[VOCAB] __attribute__((aligned(64)));
static struct { char pad[896]; float v[VOCAB * LOGIT_STRIDE] __attribute__((aligned(64))); } CUM_POS1_LUT_STORAGE __attribute__((aligned(64)));
#define CUM_POS1_LUT CUM_POS1_LUT_STORAGE.v
static float SUM_POS1_LUT[VOCAB] __attribute__((aligned(64)));
static struct { char pad[960]; float v[VOCAB * VOCAB * LOGIT_STRIDE] __attribute__((aligned(64))); } CUM_POS2_LUT_STORAGE __attribute__((aligned(64)));
#define CUM_POS2_LUT CUM_POS2_LUT_STORAGE.v
static float SUM_POS2_LUT[VOCAB * VOCAB] __attribute__((aligned(64)));
static uint8_t MAXI_POS2_LUT[VOCAB * VOCAB] __attribute__((aligned(64)));
static struct { char pad[1024]; float v[VOCAB * VOCAB * VOCAB * LOGIT_STRIDE] __attribute__((aligned(64))); } CUM_POS3_LUT_STORAGE __attribute__((aligned(64)));
#define CUM_POS3_LUT CUM_POS3_LUT_STORAGE.v
static float SUM_POS3_LUT[VOCAB * VOCAB * VOCAB] __attribute__((aligned(64)));
static uint8_t MAXI_POS3_LUT[VOCAB * VOCAB * VOCAB] __attribute__((aligned(64)));
static struct { char pad[1088]; float v[VOCAB * VOCAB * VOCAB * VOCAB * LOGIT_STRIDE] __attribute__((aligned(64))); } CUM_POS4_LUT_STORAGE __attribute__((aligned(64)));
#define CUM_POS4_LUT CUM_POS4_LUT_STORAGE.v
static float SUM_POS4_LUT[VOCAB * VOCAB * VOCAB * VOCAB] __attribute__((aligned(64)));
static uint8_t MAXI_POS4_LUT[VOCAB * VOCAB * VOCAB * VOCAB] __attribute__((aligned(64)));
#define POS5_CUM_ENTRIES ((size_t)VOCAB * VOCAB * VOCAB * VOCAB * VOCAB)
static float *CUM_POS5_LUT;
static float *SUM_POS5_LUT;
static uint8_t *MAXI_POS5_LUT;
static float W2T[MLP_H * W2T_STRIDE] __attribute__((aligned(64)));

#define LUT_BASE(tok, pos) (((tok) * BLOCK + (pos)) * EMBD)
#define TP_INDEX(tok, pos) ((tok) * BLOCK + (pos))
#define QK_PAIR_INDEX(qtp, ktp, hi) ((((qtp) * (VOCAB * BLOCK) + (ktp)) * HEAD) + (hi))

static double now_sec(void);

HOT_INLINE uint32_t xrand(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}
HOT_INLINE float urand(uint32_t *state) { return (xrand(state) >> 8) * (1.0f / (1u << 24)); }

HOT_INLINE float strict_expf(float x) {
    return expf(x);
}

HOT_INLINE void matvec_16in(const float *restrict Wm, const float *restrict x, float *restrict y, int R) {
    float32x4_t x0 = vld1q_f32(x +  0);
    float32x4_t x1 = vld1q_f32(x +  4);
    float32x4_t x2 = vld1q_f32(x +  8);
    float32x4_t x3 = vld1q_f32(x + 12);
    for (int r = 0; r < R; r++) {
        const float *wr = Wm + r * EMBD;
        float32x4_t a = vmulq_f32(vld1q_f32(wr +  0), x0);
        a = vfmaq_f32(a, vld1q_f32(wr +  4), x1);
        a = vfmaq_f32(a, vld1q_f32(wr +  8), x2);
        a = vfmaq_f32(a, vld1q_f32(wr + 12), x3);
        y[r] = vaddvq_f32(a);
    }
}

HOT_INLINE void rmsnorm(float *x) {
    float32x4_t a = vmulq_f32(vld1q_f32(x +  0), vld1q_f32(x +  0));
    a = vfmaq_f32(a, vld1q_f32(x +  4), vld1q_f32(x +  4));
    a = vfmaq_f32(a, vld1q_f32(x +  8), vld1q_f32(x +  8));
    a = vfmaq_f32(a, vld1q_f32(x + 12), vld1q_f32(x + 12));
    float ms = vaddvq_f32(a) / EMBD;
    float scale = 1.0f / sqrtf(ms + 1e-5f);
    float32x4_t s = vdupq_n_f32(scale);
    vst1q_f32(x +  0, vmulq_f32(vld1q_f32(x +  0), s));
    vst1q_f32(x +  4, vmulq_f32(vld1q_f32(x +  4), s));
    vst1q_f32(x +  8, vmulq_f32(vld1q_f32(x +  8), s));
    vst1q_f32(x + 12, vmulq_f32(vld1q_f32(x + 12), s));
}

HOT_INLINE void copy16(float *restrict dst, const float *restrict src) {
    vst1q_f32(dst +  0, vld1q_f32(src +  0));
    vst1q_f32(dst +  4, vld1q_f32(src +  4));
    vst1q_f32(dst +  8, vld1q_f32(src +  8));
    vst1q_f32(dst + 12, vld1q_f32(src + 12));
}

HOT_INLINE void rmsnorm_save(float *x, float *save) {
    float32x4_t x0 = vld1q_f32(x +  0);
    float32x4_t x1 = vld1q_f32(x +  4);
    float32x4_t x2 = vld1q_f32(x +  8);
    float32x4_t x3 = vld1q_f32(x + 12);
    vst1q_f32(save +  0, x0);
    vst1q_f32(save +  4, x1);
    vst1q_f32(save +  8, x2);
    vst1q_f32(save + 12, x3);
    float32x4_t a = vmulq_f32(x0, x0);
    a = vfmaq_f32(a, x1, x1);
    a = vfmaq_f32(a, x2, x2);
    a = vfmaq_f32(a, x3, x3);
    float ms = vaddvq_f32(a) / EMBD;
    float32x4_t s = vdupq_n_f32(1.0f / sqrtf(ms + 1e-5f));
    vst1q_f32(x +  0, vmulq_f32(x0, s));
    vst1q_f32(x +  4, vmulq_f32(x1, s));
    vst1q_f32(x +  8, vmulq_f32(x2, s));
    vst1q_f32(x + 12, vmulq_f32(x3, s));
}

__attribute__((unused))
HOT_INLINE void rmsnorm_vecs(const float *x,
                             float32x4_t *rx0, float32x4_t *rx1,
                             float32x4_t *rx2, float32x4_t *rx3,
                             float32x4_t *nx0, float32x4_t *nx1,
                             float32x4_t *nx2, float32x4_t *nx3) {
    float32x4_t x0 = vld1q_f32(x +  0);
    float32x4_t x1 = vld1q_f32(x +  4);
    float32x4_t x2 = vld1q_f32(x +  8);
    float32x4_t x3 = vld1q_f32(x + 12);
    *rx0 = x0;
    *rx1 = x1;
    *rx2 = x2;
    *rx3 = x3;
    float32x4_t a = vmulq_f32(x0, x0);
    a = vfmaq_f32(a, x1, x1);
    a = vfmaq_f32(a, x2, x2);
    a = vfmaq_f32(a, x3, x3);
    float ms = vaddvq_f32(a) / EMBD;
    float32x4_t s = vdupq_n_f32(1.0f / sqrtf(ms + 1e-5f));
    *nx0 = vmulq_f32(x0, s);
    *nx1 = vmulq_f32(x1, s);
    *nx2 = vmulq_f32(x2, s);
    *nx3 = vmulq_f32(x3, s);
}

#define RMSNORM_VECS_LOCAL(x_, rx0_, rx1_, rx2_, rx3_, nx0_, nx1_, nx2_, nx3_) do { \
    (rx0_) = vld1q_f32((x_) +  0); \
    (rx1_) = vld1q_f32((x_) +  4); \
    (rx2_) = vld1q_f32((x_) +  8); \
    (rx3_) = vld1q_f32((x_) + 12); \
    float32x4_t rms_a_ = vmulq_f32((rx0_), (rx0_)); \
    rms_a_ = vfmaq_f32(rms_a_, (rx1_), (rx1_)); \
    rms_a_ = vfmaq_f32(rms_a_, (rx2_), (rx2_)); \
    rms_a_ = vfmaq_f32(rms_a_, (rx3_), (rx3_)); \
    float rms_ms_ = vaddvq_f32(rms_a_) / EMBD; \
    float32x4_t rms_s_ = vdupq_n_f32(1.0f / sqrtf(rms_ms_ + 1e-5f)); \
    (nx0_) = vmulq_f32((rx0_), rms_s_); \
    (nx1_) = vmulq_f32((rx1_), rms_s_); \
    (nx2_) = vmulq_f32((rx2_), rms_s_); \
    (nx3_) = vmulq_f32((rx3_), rms_s_); \
} while (0)

static void precompute_w2_transpose(void) {
    const float *w2 = W + OFF_W2;
    for (int c = 0; c < MLP_H; c++) {
        for (int r = 0; r < EMBD; r++) {
            W2T[c * W2T_STRIDE + r] = w2[r * MLP_H + c];
        }
    }



}

static void precompute_front_half(void) {
    float x[EMBD] __attribute__((aligned(64)));
    for (int tok = 0; tok < VOCAB; tok++) {
        for (int pos = 0; pos < BLOCK; pos++) {
            const float *wte = W + OFF_WTE + tok * EMBD;
            const float *wpe = W + OFF_WPE + pos * EMBD;
            int base = LUT_BASE(tok, pos);
            for (int i = 0; i < EMBD; i += 4) {
                vst1q_f32(x + i, vaddq_f32(vld1q_f32(wte + i), vld1q_f32(wpe + i)));
            }
            rmsnorm(x);
            memcpy(XR_LUT + base, x, sizeof(x));
            rmsnorm(x);
            matvec_16in(W + OFF_WQ, x, Q_LUT + base, EMBD);
            matvec_16in(W + OFF_WK, x, K_LUT + base, EMBD);
            matvec_16in(W + OFF_WV, x, V_LUT + base, EMBD);
            for (int hi = 0; hi < HEAD; hi++) {
                const float *q = Q_LUT + base + hi * HD;
                const float *k = K_LUT + base + hi * HD;
                QK_SELF_LUT[(tok * BLOCK + pos) * HEAD + hi] =
                    vaddvq_f32(vmulq_f32(vld1q_f32(q), vld1q_f32(k))) * 0.5f;
            }
        }
    }


    for (int qtok = 0; qtok < VOCAB; qtok++) {
        for (int qpos = 0; qpos < BLOCK; qpos++) {
            int qtp = TP_INDEX(qtok, qpos);
            const float *qbase = Q_LUT + LUT_BASE(qtok, qpos);
            for (int ktok = 0; ktok < VOCAB; ktok++) {
                for (int kpos = 0; kpos < BLOCK; kpos++) {
                    int ktp = TP_INDEX(ktok, kpos);
                    const float *kbase = K_LUT + LUT_BASE(ktok, kpos);
                    for (int hi = 0; hi < HEAD; hi++) {
                        const float *q = qbase + hi * HD;
                        const float *k = kbase + hi * HD;
                        QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp, hi)] =
                            vaddvq_f32(vmulq_f32(vld1q_f32(q), vld1q_f32(k))) * 0.5f;
                    }
                }
            }
        }
    }

    for (int tok = 0; tok < VOCAB; tok++) {
        const float *in = V_LUT + LUT_BASE(tok, 0);
        const float *resid = XR_LUT + LUT_BASE(tok, 0);
        float *out = X_POS0_LUT + tok * EMBD;
        matvec_16in(W + OFF_WO, in, out, EMBD);
        for (int i = 0; i < EMBD; i++) {
            out[i] += resid[i];
        }
    }

    const float *k0 = K_LUT + LUT_BASE(BOS, 0);
    const float *v0 = V_LUT + LUT_BASE(BOS, 0);
    for (int tok = 0; tok < VOCAB; tok++) {
        int base = LUT_BASE(tok, 1);
        const float *q = Q_LUT + base;
        const float *cur_v = V_LUT + base;
        float head_out[EMBD] __attribute__((aligned(64)));
        for (int hi = 0; hi < HEAD; hi++) {
            const float *qh = q + hi * HD;
            float32x4_t qv = vld1q_f32(qh);
            float al0 = vaddvq_f32(vmulq_f32(qv, vld1q_f32(k0 + hi * HD))) * 0.5f;
            float al1 = QK_SELF_LUT[(tok * BLOCK + 1) * HEAD + hi];
            float maxl = (al0 > al1) ? al0 : al1;
            al0 = (al0 == maxl) ? 1.0f : strict_expf(al0 - maxl);
            al1 = (al1 == maxl) ? 1.0f : strict_expf(al1 - maxl);
            float inv = 1.0f / (al0 + al1);
            float32x4_t out = vdupq_n_f32(0.0f);
            out = vfmaq_n_f32(out, vld1q_f32(v0 + hi * HD), al0 * inv);
            out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al1 * inv);
            vst1q_f32(head_out + hi * HD, out);
        }
        float *out = X_POS1_LUT + tok * EMBD;
        matvec_16in(W + OFF_WO, head_out, out, EMBD);
        const float *resid = XR_LUT + base;
        for (int i = 0; i < EMBD; i++) {
            out[i] += resid[i];
        }
    }

}

HOT_INLINE void wo_project_add_vecs(float32x4_t x0, float32x4_t x1, float32x4_t x2, float32x4_t x3,
                                    const float *restrict resid, float *restrict out) {
    for (int r = 0; r < EMBD; r++) {
        const float *wr = W + OFF_WO + r * EMBD;
        float32x4_t a = vmulq_f32(vld1q_f32(wr +  0), x0);
        a = vfmaq_f32(a, vld1q_f32(wr +  4), x1);
        a = vfmaq_f32(a, vld1q_f32(wr +  8), x2);
        a = vfmaq_f32(a, vld1q_f32(wr + 12), x3);
        out[r] = vaddvq_f32(a) + resid[r];
    }
}

HOT_INLINE void attention_generic(int pos, int base, const float *restrict cur_v,
                                  const float *restrict K, const float *restrict V,
                                  const STEP_T_TYPE *restrict T,
                                  const float *restrict xr0, float *restrict x,
                                  float *restrict al) {
    float32x4_t hv0, hv1, hv2, hv3;
    int t_n = pos + 1;
    int qk_base = (base >> 4) * HEAD;
    int qtp = base / EMBD;
    const float *restrict qk_pair_qbase = QK_PAIR_LUT + (size_t)qtp * (VOCAB * BLOCK * HEAD);
    for (int hi = 0; hi < HEAD; hi++) {
        const float *restrict qk_pair_head = qk_pair_qbase + hi;
        float maxl = -1e30f;
        for (int t = 0; t < pos; t++) {
            al[t] = qk_pair_head[STEP_TP(T, t) * HEAD];
            if (al[t] > maxl) maxl = al[t];
        }
        {
            al[pos] = QK_SELF_LUT[qk_base + hi];
            if (al[pos] > maxl) maxl = al[pos];
        }
        float sum = 0.0f;
        for (int t = 0; t < t_n; t++) {
            al[t] = (al[t] == maxl) ? 1.0f : strict_expf(al[t] - maxl);
            sum += al[t];
        }
        float inv = 1.0f / sum;
        float32x4_t out = vdupq_n_f32(0.0f);
        for (int t = 0; t < pos; t++) {
            float w = al[t] * inv;
            const float *vh = STEP_V_PTR(T, V, t, hi);
            out = vfmaq_n_f32(out, vld1q_f32(vh), w);
        }
        out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al[pos] * inv);
        if (hi == 0) hv0 = out;
        else if (hi == 1) hv1 = out;
        else if (hi == 2) hv2 = out;
        else hv3 = out;
    }
    wo_project_add_vecs(hv0, hv1, hv2, hv3, xr0, x);
}

HOT_INLINE void mlp_fused_add_vecs_out(float32x4_t x0, float32x4_t x1, float32x4_t x2, float32x4_t x3,
                                       float32x4_t r0, float32x4_t r1, float32x4_t r2, float32x4_t r3,
                                       float32x4_t *restrict y0, float32x4_t *restrict y1,
                                       float32x4_t *restrict y2, float32x4_t *restrict y3) {
    float32x4_t o0 = vdupq_n_f32(0.0f);
    float32x4_t o1 = vdupq_n_f32(0.0f);
    float32x4_t o2 = vdupq_n_f32(0.0f);
    float32x4_t o3 = vdupq_n_f32(0.0f);

    #pragma clang loop unroll(full)
    for (int i = 0; i < MLP_H; i += 4) {
        const float *w10 = W + OFF_W1 + i * EMBD;
        const float *w11 = w10 + EMBD;
        const float *w12 = w11 + EMBD;
        const float *w13 = w12 + EMBD;
        if (i + STRICT_MLP_W1_PREFETCH_DIST < MLP_H) {
            __builtin_prefetch(W + OFF_W1 + (i + STRICT_MLP_W1_PREFETCH_DIST) * EMBD,
                               0, STRICT_MLP_W1_PREFETCH_LOCALITY);
        }
        if (i + STRICT_MLP_W2T_PREFETCH_DIST < MLP_H) {
            __builtin_prefetch(W2T + (i + STRICT_MLP_W2T_PREFETCH_DIST) * W2T_STRIDE,
                               0, STRICT_MLP_W2T_PREFETCH_LOCALITY);
        }
        float32x4_t a0 = vmulq_f32(vld1q_f32(w10 +  0), x0);
        float32x4_t a1 = vmulq_f32(vld1q_f32(w11 +  0), x0);
        float32x4_t a2 = vmulq_f32(vld1q_f32(w12 +  0), x0);
        float32x4_t a3 = vmulq_f32(vld1q_f32(w13 +  0), x0);
        a0 = vfmaq_f32(a0, vld1q_f32(w10 +  4), x1);
        a1 = vfmaq_f32(a1, vld1q_f32(w11 +  4), x1);
        a2 = vfmaq_f32(a2, vld1q_f32(w12 +  4), x1);
        a3 = vfmaq_f32(a3, vld1q_f32(w13 +  4), x1);
        a0 = vfmaq_f32(a0, vld1q_f32(w10 +  8), x2);
        a1 = vfmaq_f32(a1, vld1q_f32(w11 +  8), x2);
        a2 = vfmaq_f32(a2, vld1q_f32(w12 +  8), x2);
        a3 = vfmaq_f32(a3, vld1q_f32(w13 +  8), x2);
        a0 = vfmaq_f32(a0, vld1q_f32(w10 + 12), x3);
        a1 = vfmaq_f32(a1, vld1q_f32(w11 + 12), x3);
        a2 = vfmaq_f32(a2, vld1q_f32(w12 + 12), x3);
        a3 = vfmaq_f32(a3, vld1q_f32(w13 + 12), x3);
        float h0 = vaddvq_f32(a0);
        float h1 = vaddvq_f32(a1);
        float h2 = vaddvq_f32(a2);
        float h3 = vaddvq_f32(a3);
        if (MLP_POSITIVE_ROW(i + 0, h0)) {
            float32x4_t hv = vdupq_n_f32(h0);
            const float *w2 = W2T + i * W2T_STRIDE;
            o0 = vfmaq_f32(o0, vld1q_f32(w2 +  0), hv);
            o1 = vfmaq_f32(o1, vld1q_f32(w2 +  4), hv);
            o2 = vfmaq_f32(o2, vld1q_f32(w2 +  8), hv);
            o3 = vfmaq_f32(o3, vld1q_f32(w2 + 12), hv);
        }
        if (MLP_POSITIVE_ROW(i + 1, h1)) {
            float32x4_t hv = vdupq_n_f32(h1);
            const float *w2 = W2T + (i + 1) * W2T_STRIDE;
            o0 = vfmaq_f32(o0, vld1q_f32(w2 +  0), hv);
            o1 = vfmaq_f32(o1, vld1q_f32(w2 +  4), hv);
            o2 = vfmaq_f32(o2, vld1q_f32(w2 +  8), hv);
            o3 = vfmaq_f32(o3, vld1q_f32(w2 + 12), hv);
        }
        if (MLP_POSITIVE_ROW(i + 2, h2)) {
            float32x4_t hv = vdupq_n_f32(h2);
            const float *w2 = W2T + (i + 2) * W2T_STRIDE;
            o0 = vfmaq_f32(o0, vld1q_f32(w2 +  0), hv);
            o1 = vfmaq_f32(o1, vld1q_f32(w2 +  4), hv);
            o2 = vfmaq_f32(o2, vld1q_f32(w2 +  8), hv);
            o3 = vfmaq_f32(o3, vld1q_f32(w2 + 12), hv);
        }
        if (MLP_POSITIVE_ROW(i + 3, h3)) {
            float32x4_t hv = vdupq_n_f32(h3);
            const float *w2 = W2T + (i + 3) * W2T_STRIDE;
            o0 = vfmaq_f32(o0, vld1q_f32(w2 +  0), hv);
            o1 = vfmaq_f32(o1, vld1q_f32(w2 +  4), hv);
            o2 = vfmaq_f32(o2, vld1q_f32(w2 +  8), hv);
            o3 = vfmaq_f32(o3, vld1q_f32(w2 + 12), hv);
        }
    }

    *y0 = vaddq_f32(o0, r0);
    *y1 = vaddq_f32(o1, r1);
    *y2 = vaddq_f32(o2, r2);
    *y3 = vaddq_f32(o3, r3);
}

HOT_INLINE void mlp_fused_add_vecs(float32x4_t x0, float32x4_t x1, float32x4_t x2, float32x4_t x3,
                                   float32x4_t r0, float32x4_t r1, float32x4_t r2, float32x4_t r3,
                                   float *restrict out) {
    float32x4_t y0, y1, y2, y3;
    mlp_fused_add_vecs_out(x0, x1, x2, x3, r0, r1, r2, r3, &y0, &y1, &y2, &y3);
    vst1q_f32(out +  0, y0);
    vst1q_f32(out +  4, y1);
    vst1q_f32(out +  8, y2);
    vst1q_f32(out + 12, y3);
}


HOT_INLINE float lm_logits_vecs(float32x4_t x0, float32x4_t x1, float32x4_t x2, float32x4_t x3,
                                float *restrict logits, int *restrict max_idx) {
    float maxl = -1e30f;
    int maxi = 0;
    int r = 0;
    for (; r + 1 < VOCAB; r += 2) {
        const float *wr = W + OFF_LM + r * EMBD;
        const float *wr1 = wr + EMBD;
        float32x4_t a0 = vmulq_f32(vld1q_f32(wr +  0), x0);
        float32x4_t a1 = vmulq_f32(vld1q_f32(wr1 + 0), x0);
        a0 = vfmaq_f32(a0, vld1q_f32(wr +  4), x1);
        a1 = vfmaq_f32(a1, vld1q_f32(wr1 + 4), x1);
        a0 = vfmaq_f32(a0, vld1q_f32(wr +  8), x2);
        a1 = vfmaq_f32(a1, vld1q_f32(wr1 + 8), x2);
        a0 = vfmaq_f32(a0, vld1q_f32(wr + 12), x3);
        a1 = vfmaq_f32(a1, vld1q_f32(wr1 + 12), x3);
        float v0 = vaddvq_f32(a0);
        float v1 = vaddvq_f32(a1);
        v0 *= INV_TEMP;
        v1 *= INV_TEMP;
        logits[r] = v0;
        if (v0 > maxl) { maxl = v0; maxi = r; }
        logits[r + 1] = v1;
        if (v1 > maxl) { maxl = v1; maxi = r + 1; }
    }
    if (r < VOCAB) {
        const float *wr = W + OFF_LM + r * EMBD;
        float32x4_t a = vmulq_f32(vld1q_f32(wr + 0), x0);
        a = vfmaq_f32(a, vld1q_f32(wr + 4), x1);
        a = vfmaq_f32(a, vld1q_f32(wr + 8), x2);
        a = vfmaq_f32(a, vld1q_f32(wr + 12), x3);
        float v = vaddvq_f32(a);
        v *= INV_TEMP;
        logits[r] = v;
        if (v > maxl) { maxl = v; maxi = r; }
    }
    *max_idx = maxi;
    return maxl;
}

HOT_INLINE float lm_logits(const float *restrict x, float *restrict logits, int *restrict max_idx) {
    float32x4_t x0 = vld1q_f32(x +  0);
    float32x4_t x1 = vld1q_f32(x +  4);
    float32x4_t x2 = vld1q_f32(x +  8);
    float32x4_t x3 = vld1q_f32(x + 12);
    return lm_logits_vecs(x0, x1, x2, x3, logits, max_idx);
}



static inline void wo_project_add_ref(const float *in, const float *resid, float *out) {
    float32x4_t x0 = vld1q_f32(in +  0);
    float32x4_t x1 = vld1q_f32(in +  4);
    float32x4_t x2 = vld1q_f32(in +  8);
    float32x4_t x3 = vld1q_f32(in + 12);
    for (int r = 0; r < EMBD; r++) {
        const float *wr = W + OFF_WO + r * EMBD;
        float32x4_t a = vmulq_f32(vld1q_f32(wr +  0), x0);
        a = vfmaq_f32(a, vld1q_f32(wr +  4), x1);
        a = vfmaq_f32(a, vld1q_f32(wr +  8), x2);
        a = vfmaq_f32(a, vld1q_f32(wr + 12), x3);
        out[r] = vaddvq_f32(a) + resid[r];
    }
}

static inline void mlp_fused_add_ref(const float *in, const float *resid, float *out) {
    float32x4_t x0 = vld1q_f32(in +  0);
    float32x4_t x1 = vld1q_f32(in +  4);
    float32x4_t x2 = vld1q_f32(in +  8);
    float32x4_t x3 = vld1q_f32(in + 12);
    float32x4_t o0 = vdupq_n_f32(0.0f);
    float32x4_t o1 = vdupq_n_f32(0.0f);
    float32x4_t o2 = vdupq_n_f32(0.0f);
    float32x4_t o3 = vdupq_n_f32(0.0f);

    for (int i = 0; i < MLP_H; i++) {
        const float *w1 = W + OFF_W1 + i * EMBD;
        float32x4_t a = vmulq_f32(vld1q_f32(w1 +  0), x0);
        a = vfmaq_f32(a, vld1q_f32(w1 +  4), x1);
        a = vfmaq_f32(a, vld1q_f32(w1 +  8), x2);
        a = vfmaq_f32(a, vld1q_f32(w1 + 12), x3);
        float h = vaddvq_f32(a);
        if (MLP_POSITIVE(h)) {
            float32x4_t hv = vdupq_n_f32(h);
            const float *w2 = W2T + i * W2T_STRIDE;
            o0 = vfmaq_f32(o0, vld1q_f32(w2 +  0), hv);
            o1 = vfmaq_f32(o1, vld1q_f32(w2 +  4), hv);
            o2 = vfmaq_f32(o2, vld1q_f32(w2 +  8), hv);
            o3 = vfmaq_f32(o3, vld1q_f32(w2 + 12), hv);
        }
    }

    vst1q_f32(out +  0, vaddq_f32(o0, vld1q_f32(resid +  0)));
    vst1q_f32(out +  4, vaddq_f32(o1, vld1q_f32(resid +  4)));
    vst1q_f32(out +  8, vaddq_f32(o2, vld1q_f32(resid +  8)));
    vst1q_f32(out + 12, vaddq_f32(o3, vld1q_f32(resid + 12)));
}

static float build_cumulative_from_logits(const float *restrict src, float maxl, int max_idx,
                                          float *restrict cum) {
    float sum = 0.0f;
    for (int i = 0; i < VOCAB - 1; i++) {
        sum += (i == max_idx) ? 1.0f : strict_expf(src[i] - maxl);
        cum[i] = sum;
    }
    sum += (max_idx == VOCAB - 1) ? 1.0f : strict_expf(src[VOCAB - 1] - maxl);
    return sum;
}

static void precompute_pos01_mlp_luts(void) {
    float n[EMBD] __attribute__((aligned(64)));

    for (int tok = 0; tok < VOCAB; tok++) {
        const float *src0 = X_POS0_LUT + tok * EMBD;
        copy16(n, src0);
        rmsnorm(n);
        float *dst0 = X2_POS0_LUT + tok * EMBD;
        mlp_fused_add_ref(n, src0, dst0);
        int maxi = 0;
        MAXL_POS0_LUT[tok] = lm_logits(dst0, LOGITS_POS0_LUT + tok * LOGIT_STRIDE, &maxi);
        MAXI_POS0_LUT[tok] = (uint8_t)maxi;
        SUM_POS0_LUT[tok] = build_cumulative_from_logits(LOGITS_POS0_LUT + tok * LOGIT_STRIDE,
                                                         MAXL_POS0_LUT[tok],
                                                         maxi,
                                                         CUM_POS0_LUT + tok * LOGIT_STRIDE);
    }

    for (int tok = 0; tok < VOCAB; tok++) {
        const float *src1 = X_POS1_LUT + tok * EMBD;
        copy16(n, src1);
        rmsnorm(n);
        float *dst1 = X2_POS1_LUT + tok * EMBD;
        mlp_fused_add_ref(n, src1, dst1);
        int maxi = 0;
        MAXL_POS1_LUT[tok] = lm_logits(dst1, LOGITS_POS1_LUT + tok * LOGIT_STRIDE, &maxi);
        MAXI_POS1_LUT[tok] = (uint8_t)maxi;
        SUM_POS1_LUT[tok] = build_cumulative_from_logits(LOGITS_POS1_LUT + tok * LOGIT_STRIDE,
                                                         MAXL_POS1_LUT[tok],
                                                         maxi,
                                                         CUM_POS1_LUT + tok * LOGIT_STRIDE);
    }
}

static void precompute_pos2_cum_luts(void) {
    float x[EMBD] __attribute__((aligned(64)));
    float n[EMBD] __attribute__((aligned(64)));
    float logits[LOGIT_STRIDE] __attribute__((aligned(64)));
    const int ktp0 = TP_INDEX(BOS, 0);
    const float *v0 = V_LUT + LUT_BASE(BOS, 0);

    for (int prev = 0; prev < VOCAB; prev++) {
        const int ktp1 = TP_INDEX(prev, 1);
        const float *v1 = V_LUT + LUT_BASE(prev, 1);
        for (int tok = 0; tok < VOCAB; tok++) {
            const int idx = prev * VOCAB + tok;
            const int base = LUT_BASE(tok, 2);
            const int qtp = TP_INDEX(tok, 2);
            const int qk_base = (base >> 4) * HEAD;
            const float *cur_v = V_LUT + base;
            const float *xr0 = XR_LUT + base;
            float32x4_t hv0, hv1, hv2, hv3;

            for (int hi = 0; hi < HEAD; hi++) {
                float al0 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp0, hi)];
                float al1 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp1, hi)];
                float al2 = QK_SELF_LUT[qk_base + hi];
                float maxl = al0;
                if (al1 > maxl) maxl = al1;
                if (al2 > maxl) maxl = al2;
                al0 = (al0 == maxl) ? 1.0f : strict_expf(al0 - maxl);
                al1 = (al1 == maxl) ? 1.0f : strict_expf(al1 - maxl);
                al2 = (al2 == maxl) ? 1.0f : strict_expf(al2 - maxl);
                float inv = 1.0f / ((al0 + al1) + al2);
                float32x4_t out = vdupq_n_f32(0.0f);
                out = vfmaq_n_f32(out, vld1q_f32(v0 + hi * HD), al0 * inv);
                out = vfmaq_n_f32(out, vld1q_f32(v1 + hi * HD), al1 * inv);
                out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al2 * inv);
                if (hi == 0) hv0 = out;
                else if (hi == 1) hv1 = out;
                else if (hi == 2) hv2 = out;
                else hv3 = out;
            }

            wo_project_add_vecs(hv0, hv1, hv2, hv3, xr0, x);
            copy16(n, x);
            rmsnorm(n);
            mlp_fused_add_ref(n, x, x);

            int maxi = 0;
            float maxl = lm_logits(x, logits, &maxi);
            MAXI_POS2_LUT[idx] = (uint8_t)maxi;
            SUM_POS2_LUT[idx] = build_cumulative_from_logits(logits, maxl, maxi,
                                                             CUM_POS2_LUT + idx * LOGIT_STRIDE);
        }
    }
}

static void precompute_pos3_cum_luts(void) {
    float x[EMBD] __attribute__((aligned(64)));
    float n[EMBD] __attribute__((aligned(64)));
    float logits[LOGIT_STRIDE] __attribute__((aligned(64)));
    const int ktp0 = TP_INDEX(BOS, 0);
    const float *v0 = V_LUT + LUT_BASE(BOS, 0);

    for (int prev1 = 0; prev1 < VOCAB; prev1++) {
        const int ktp1 = TP_INDEX(prev1, 1);
        const float *v1 = V_LUT + LUT_BASE(prev1, 1);
        for (int prev2 = 0; prev2 < VOCAB; prev2++) {
            const int ktp2 = TP_INDEX(prev2, 2);
            const float *v2 = V_LUT + LUT_BASE(prev2, 2);
            for (int tok = 0; tok < VOCAB; tok++) {
                const int idx = (prev1 * VOCAB + prev2) * VOCAB + tok;
                const int base = LUT_BASE(tok, 3);
                const int qtp = TP_INDEX(tok, 3);
                const int qk_base = (base >> 4) * HEAD;
                const float *cur_v = V_LUT + base;
                const float *xr0 = XR_LUT + base;
                float32x4_t hv0, hv1, hv2, hv3;

                for (int hi = 0; hi < HEAD; hi++) {
                    float al0 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp0, hi)];
                    float al1 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp1, hi)];
                    float al2 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp2, hi)];
                    float al3 = QK_SELF_LUT[qk_base + hi];
                    float maxl = al0;
                    if (al1 > maxl) maxl = al1;
                    if (al2 > maxl) maxl = al2;
                    if (al3 > maxl) maxl = al3;
                    al0 = (al0 == maxl) ? 1.0f : strict_expf(al0 - maxl);
                    al1 = (al1 == maxl) ? 1.0f : strict_expf(al1 - maxl);
                    al2 = (al2 == maxl) ? 1.0f : strict_expf(al2 - maxl);
                    al3 = (al3 == maxl) ? 1.0f : strict_expf(al3 - maxl);
                    float inv = 1.0f / (((al0 + al1) + al2) + al3);
                    float32x4_t out = vdupq_n_f32(0.0f);
                    out = vfmaq_n_f32(out, vld1q_f32(v0 + hi * HD), al0 * inv);
                    out = vfmaq_n_f32(out, vld1q_f32(v1 + hi * HD), al1 * inv);
                    out = vfmaq_n_f32(out, vld1q_f32(v2 + hi * HD), al2 * inv);
                    out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al3 * inv);
                    if (hi == 0) hv0 = out;
                    else if (hi == 1) hv1 = out;
                    else if (hi == 2) hv2 = out;
                    else hv3 = out;
                }

                wo_project_add_vecs(hv0, hv1, hv2, hv3, xr0, x);
                copy16(n, x);
                rmsnorm(n);
                mlp_fused_add_ref(n, x, x);

                int maxi = 0;
                float maxl = lm_logits(x, logits, &maxi);
                MAXI_POS3_LUT[idx] = (uint8_t)maxi;
                SUM_POS3_LUT[idx] = build_cumulative_from_logits(logits, maxl, maxi,
                                                                 CUM_POS3_LUT + idx * LOGIT_STRIDE);
            }
        }
    }
}

static void precompute_pos4_cum_luts(void) {
    float x[EMBD] __attribute__((aligned(64)));
    float n[EMBD] __attribute__((aligned(64)));
    float logits[LOGIT_STRIDE] __attribute__((aligned(64)));
    const int ktp0 = TP_INDEX(BOS, 0);
    const float *v0 = V_LUT + LUT_BASE(BOS, 0);

    for (int prev1 = 0; prev1 < VOCAB; prev1++) {
        const int ktp1 = TP_INDEX(prev1, 1);
        const float *v1 = V_LUT + LUT_BASE(prev1, 1);
        for (int prev2 = 0; prev2 < VOCAB; prev2++) {
            const int ktp2 = TP_INDEX(prev2, 2);
            const float *v2 = V_LUT + LUT_BASE(prev2, 2);
            for (int prev3 = 0; prev3 < VOCAB; prev3++) {
                const int ktp3 = TP_INDEX(prev3, 3);
                const float *v3 = V_LUT + LUT_BASE(prev3, 3);
                for (int tok = 0; tok < VOCAB; tok++) {
                    const int idx = ((prev1 * VOCAB + prev2) * VOCAB + prev3) * VOCAB + tok;
                    const int base = LUT_BASE(tok, 4);
                    const int qtp = TP_INDEX(tok, 4);
                    const int qk_base = (base >> 4) * HEAD;
                    const float *cur_v = V_LUT + base;
                    const float *xr0 = XR_LUT + base;
                    float32x4_t hv0, hv1, hv2, hv3;

                    for (int hi = 0; hi < HEAD; hi++) {
                        float al0 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp0, hi)];
                        float al1 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp1, hi)];
                        float al2 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp2, hi)];
                        float al3 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp3, hi)];
                        float al4 = QK_SELF_LUT[qk_base + hi];
                        float maxl = al0;
                        if (al1 > maxl) maxl = al1;
                        if (al2 > maxl) maxl = al2;
                        if (al3 > maxl) maxl = al3;
                        if (al4 > maxl) maxl = al4;
                        al0 = (al0 == maxl) ? 1.0f : strict_expf(al0 - maxl);
                        al1 = (al1 == maxl) ? 1.0f : strict_expf(al1 - maxl);
                        al2 = (al2 == maxl) ? 1.0f : strict_expf(al2 - maxl);
                        al3 = (al3 == maxl) ? 1.0f : strict_expf(al3 - maxl);
                        al4 = (al4 == maxl) ? 1.0f : strict_expf(al4 - maxl);
                        float inv = 1.0f / ((((al0 + al1) + al2) + al3) + al4);
                        float32x4_t out = vdupq_n_f32(0.0f);
                        out = vfmaq_n_f32(out, vld1q_f32(v0 + hi * HD), al0 * inv);
                        out = vfmaq_n_f32(out, vld1q_f32(v1 + hi * HD), al1 * inv);
                        out = vfmaq_n_f32(out, vld1q_f32(v2 + hi * HD), al2 * inv);
                        out = vfmaq_n_f32(out, vld1q_f32(v3 + hi * HD), al3 * inv);
                        out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al4 * inv);
                        if (hi == 0) hv0 = out;
                        else if (hi == 1) hv1 = out;
                        else if (hi == 2) hv2 = out;
                        else hv3 = out;
                    }

                    wo_project_add_vecs(hv0, hv1, hv2, hv3, xr0, x);
                    copy16(n, x);
                    rmsnorm(n);
                    mlp_fused_add_ref(n, x, x);

                    int maxi = 0;
                    float maxl = lm_logits(x, logits, &maxi);
                    MAXI_POS4_LUT[idx] = (uint8_t)maxi;
                    SUM_POS4_LUT[idx] = build_cumulative_from_logits(logits, maxl, maxi,
                                                                     CUM_POS4_LUT + idx * LOGIT_STRIDE);
                }
            }
        }
    }
}

static void *alloc_pos5_table(size_t bytes, const char *name) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0 || ptr == NULL) {
        fprintf(stderr, "failed to allocate %s (%zu bytes)\n", name, bytes);
        exit(1);
    }
    return ptr;
}

static void precompute_pos5_cum_luts(void) {
    float x[EMBD] __attribute__((aligned(64)));
    float n[EMBD] __attribute__((aligned(64)));
    float logits[LOGIT_STRIDE] __attribute__((aligned(64)));
    const int ktp0 = TP_INDEX(BOS, 0);
    const float *v0 = V_LUT + LUT_BASE(BOS, 0);

    if (CUM_POS5_LUT == NULL) {
        CUM_POS5_LUT = (float *)alloc_pos5_table(POS5_CUM_ENTRIES * LOGIT_STRIDE * sizeof(float),
                                                 "CUM_POS5_LUT");
        SUM_POS5_LUT = (float *)alloc_pos5_table(POS5_CUM_ENTRIES * sizeof(float),
                                                 "SUM_POS5_LUT");
        MAXI_POS5_LUT = (uint8_t *)alloc_pos5_table(POS5_CUM_ENTRIES * sizeof(uint8_t),
                                                    "MAXI_POS5_LUT");
    }

    for (int prev1 = 0; prev1 < VOCAB; prev1++) {
        const int ktp1 = TP_INDEX(prev1, 1);
        const float *v1 = V_LUT + LUT_BASE(prev1, 1);
        for (int prev2 = 0; prev2 < VOCAB; prev2++) {
            const int ktp2 = TP_INDEX(prev2, 2);
            const float *v2 = V_LUT + LUT_BASE(prev2, 2);
            for (int prev3 = 0; prev3 < VOCAB; prev3++) {
                const int ktp3 = TP_INDEX(prev3, 3);
                const float *v3 = V_LUT + LUT_BASE(prev3, 3);
                for (int prev4 = 0; prev4 < VOCAB; prev4++) {
                    const int ktp4 = TP_INDEX(prev4, 4);
                    const float *v4 = V_LUT + LUT_BASE(prev4, 4);
                    for (int tok = 0; tok < VOCAB; tok++) {
                        const size_t idx = ((((size_t)prev1 * VOCAB + prev2) * VOCAB + prev3) * VOCAB + prev4) * VOCAB + tok;
                        const int base = LUT_BASE(tok, 5);
                        const int qtp = TP_INDEX(tok, 5);
                        const int qk_base = (base >> 4) * HEAD;
                        const float *cur_v = V_LUT + base;
                        const float *xr0 = XR_LUT + base;
                        float32x4_t hv0, hv1, hv2, hv3;

                        for (int hi = 0; hi < HEAD; hi++) {
                            float al0 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp0, hi)];
                            float al1 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp1, hi)];
                            float al2 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp2, hi)];
                            float al3 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp3, hi)];
                            float al4 = QK_PAIR_LUT[QK_PAIR_INDEX(qtp, ktp4, hi)];
                            float al5 = QK_SELF_LUT[qk_base + hi];
                            float maxl = al0;
                            if (al1 > maxl) maxl = al1;
                            if (al2 > maxl) maxl = al2;
                            if (al3 > maxl) maxl = al3;
                            if (al4 > maxl) maxl = al4;
                            if (al5 > maxl) maxl = al5;
                            al0 = (al0 == maxl) ? 1.0f : strict_expf(al0 - maxl);
                            al1 = (al1 == maxl) ? 1.0f : strict_expf(al1 - maxl);
                            al2 = (al2 == maxl) ? 1.0f : strict_expf(al2 - maxl);
                            al3 = (al3 == maxl) ? 1.0f : strict_expf(al3 - maxl);
                            al4 = (al4 == maxl) ? 1.0f : strict_expf(al4 - maxl);
                            al5 = (al5 == maxl) ? 1.0f : strict_expf(al5 - maxl);
                            float inv = 1.0f / (((((al0 + al1) + al2) + al3) + al4) + al5);
                            float32x4_t out = vdupq_n_f32(0.0f);
                            out = vfmaq_n_f32(out, vld1q_f32(v0 + hi * HD), al0 * inv);
                            out = vfmaq_n_f32(out, vld1q_f32(v1 + hi * HD), al1 * inv);
                            out = vfmaq_n_f32(out, vld1q_f32(v2 + hi * HD), al2 * inv);
                            out = vfmaq_n_f32(out, vld1q_f32(v3 + hi * HD), al3 * inv);
                            out = vfmaq_n_f32(out, vld1q_f32(v4 + hi * HD), al4 * inv);
                            out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al5 * inv);
                            if (hi == 0) hv0 = out;
                            else if (hi == 1) hv1 = out;
                            else if (hi == 2) hv2 = out;
                            else hv3 = out;
                        }

                        wo_project_add_vecs(hv0, hv1, hv2, hv3, xr0, x);
                        copy16(n, x);
                        rmsnorm(n);
                        mlp_fused_add_ref(n, x, x);

                        int maxi = 0;
                        float maxl = lm_logits(x, logits, &maxi);
                        MAXI_POS5_LUT[idx] = (uint8_t)maxi;
                        SUM_POS5_LUT[idx] = build_cumulative_from_logits(logits, maxl, maxi,
                                                                         CUM_POS5_LUT + idx * LOGIT_STRIDE);
                    }
                }
            }
        }
    }
}

static inline float lm_logits_ref(const float *x, float *logits) {
    float32x4_t x0 = vld1q_f32(x +  0);
    float32x4_t x1 = vld1q_f32(x +  4);
    float32x4_t x2 = vld1q_f32(x +  8);
    float32x4_t x3 = vld1q_f32(x + 12);
    float maxl = -1e30f;
    for (int r = 0; r < VOCAB; r++) {
        const float *wr = W + OFF_LM + r * EMBD;
        float32x4_t a = vmulq_f32(vld1q_f32(wr +  0), x0);
        a = vfmaq_f32(a, vld1q_f32(wr +  4), x1);
        a = vfmaq_f32(a, vld1q_f32(wr +  8), x2);
        a = vfmaq_f32(a, vld1q_f32(wr + 12), x3);
        float v = vaddvq_f32(a) * INV_TEMP;
        logits[r] = v;
        if (v > maxl) maxl = v;
    }
    return maxl;
}

HOT_INLINE int sample_weights(const float *p, float sum, uint32_t *rng) {
    float r = urand(rng) * sum;
    float c = 0.0f;
    for (int i = 0; i < VOCAB - 1; i++) {
        c += p[i];
        if (r < c) return i;
    }
    return VOCAB - 1;
}

HOT_INLINE int sample_cumulative(const float *cum, float sum, uint32_t *rng) {
    float r = urand(rng) * sum;
    for (int i = 0; i < VOCAB - 1; i++) {
        if (r < cum[i]) return i;
    }
    return VOCAB - 1;
}

#define SAMPLE_CUM_POS01_I(i_) do { if (r < cum[(i_)]) return (i_); } while (0)
HOT_INLINE int sample_cumulative_pos01_shape(const float *cum, float sum, uint32_t *rng) {
    float r = urand(rng) * sum;
    SAMPLE_CUM_POS01_I(0);
    SAMPLE_CUM_POS01_I(1);
    SAMPLE_CUM_POS01_I(2);
    SAMPLE_CUM_POS01_I(3);
    SAMPLE_CUM_POS01_I(4);
    SAMPLE_CUM_POS01_I(5);
    SAMPLE_CUM_POS01_I(6);
    SAMPLE_CUM_POS01_I(7);
    SAMPLE_CUM_POS01_I(8);
    SAMPLE_CUM_POS01_I(9);
    SAMPLE_CUM_POS01_I(10);
    SAMPLE_CUM_POS01_I(11);
    SAMPLE_CUM_POS01_I(12);
    SAMPLE_CUM_POS01_I(13);
    SAMPLE_CUM_POS01_I(14);
    SAMPLE_CUM_POS01_I(15);
    SAMPLE_CUM_POS01_I(16);
    SAMPLE_CUM_POS01_I(17);
    SAMPLE_CUM_POS01_I(18);
    SAMPLE_CUM_POS01_I(19);
    SAMPLE_CUM_POS01_I(20);
    SAMPLE_CUM_POS01_I(21);
    SAMPLE_CUM_POS01_I(22);
    SAMPLE_CUM_POS01_I(23);
    SAMPLE_CUM_POS01_I(24);
    SAMPLE_CUM_POS01_I(25);
    return VOCAB - 1;
}
#undef SAMPLE_CUM_POS01_I

HOT_INLINE int sample_cumulative_neon(const float *cum, float sum, uint32_t *rng) {
    float r = urand(rng) * sum;
    float32x4_t rv = vdupq_n_f32(r);
#pragma clang loop unroll(full)
    for (int i = 0; i < 24; i += 4) {
        uint32x4_t m = vcgtq_f32(vld1q_f32(cum + i), rv);
        uint64x2_t mu = vreinterpretq_u64_u32(m);
        uint64_t lo = vgetq_lane_u64(mu, 0);
        uint64_t hi = vgetq_lane_u64(mu, 1);
        if (lo | hi) {
            if ((uint32_t)lo) return i;
            if (lo >> 32) return i + 1;
            if ((uint32_t)hi) return i + 2;
            return i + 3;
        }
    }
    if (r < cum[24]) return 24;
    if (r < cum[25]) return 25;
    return VOCAB - 1;
}


#define BUILD_CUM_K(K_) do { \
    float sum_ = 0.0f; \
    _Pragma("clang loop unroll(full)") \
    for (int i_ = 0; i_ < (K_); i_++) { \
        sum_ += strict_expf(src[i_] - maxl); \
        cum[i_] = sum_; \
    } \
    if ((K_) < VOCAB - 1) { \
        sum_ += 1.0f; \
        cum[(K_)] = sum_; \
        _Pragma("clang loop unroll(full)") \
        for (int i_ = (K_) + 1; i_ < VOCAB - 1; i_++) { \
            sum_ += strict_expf(src[i_] - maxl); \
            cum[i_] = sum_; \
        } \
        sum_ += strict_expf(src[VOCAB - 1] - maxl); \
    } else { \
        sum_ += 1.0f; \
    } \
    return sum_; \
} while (0)

HOT_INLINE float build_cumulative_live(const float *restrict src, float maxl,
                                       int max_idx, float *restrict cum) {
    switch (max_idx) {
        case 0: BUILD_CUM_K(0);
        case 4: BUILD_CUM_K(4);
        case 8: BUILD_CUM_K(8);
        case 11: BUILD_CUM_K(11);
        case 13: BUILD_CUM_K(13);
        case 17: BUILD_CUM_K(17);
        case 26: BUILD_CUM_K(26);
        default: {
            float sum = 0.0f;
            for (int i = 0; i < VOCAB - 1; i++) {
                sum += (i == max_idx) ? 1.0f : strict_expf(src[i] - maxl);
                cum[i] = sum;
            }
            sum += (max_idx == VOCAB - 1) ? 1.0f : strict_expf(src[VOCAB - 1] - maxl);
            return sum;
        }
    }
}
#undef BUILD_CUM_K

HOT_INLINE int step_pos0_precomputed(float *restrict K, float *restrict V STEP_T_PARAM, uint32_t *restrict rng) {
    const int base = LUT_BASE(BOS, 0);
    int nxt = sample_cumulative_pos01_shape(CUM_POS0_LUT + BOS * LOGIT_STRIDE, SUM_POS0_LUT[BOS], rng);
    if (nxt != BOS) {
        copy16(K, K_LUT + base);
        copy16(V, V_LUT + base);
        T[0] = BOS;
    }
    return nxt;
}

HOT_INLINE int step_pos1_precomputed(int tok, float *restrict K, float *restrict V STEP_T_PARAM,
                                     uint32_t *restrict rng) {
    const int base = LUT_BASE(tok, 1);
    int nxt = sample_cumulative_pos01_shape(CUM_POS1_LUT + tok * LOGIT_STRIDE, SUM_POS1_LUT[tok], rng);
    if (nxt != BOS) {
        copy16(K + EMBD, K_LUT + base);
        copy16(V + EMBD, V_LUT + base);
        T[1] = (uint8_t)tok;
    }
    return nxt;
}

HOT_INLINE int step_pos2_precomputed(int tok, float *restrict K, float *restrict V STEP_T_PARAM,
                                     uint32_t *restrict rng) {
    const int base = LUT_BASE(tok, 2);
    const int idx = (int)T[1] * VOCAB + tok;
    int nxt = sample_cumulative_pos01_shape(CUM_POS2_LUT + idx * LOGIT_STRIDE, SUM_POS2_LUT[idx], rng);
    if (nxt != BOS) {
        copy16(K + 2 * EMBD, K_LUT + base);
        copy16(V + 2 * EMBD, V_LUT + base);
        T[2] = (uint8_t)tok;
    }
    return nxt;
}

HOT_INLINE int step_pos3_precomputed(int tok, float *restrict K, float *restrict V STEP_T_PARAM,
                                     uint32_t *restrict rng) {
    const int base = LUT_BASE(tok, 3);
    const int idx = ((int)T[1] * VOCAB + (int)T[2]) * VOCAB + tok;
    int nxt = sample_cumulative_pos01_shape(CUM_POS3_LUT + idx * LOGIT_STRIDE, SUM_POS3_LUT[idx], rng);
    if (nxt != BOS) {
        copy16(K + 3 * EMBD, K_LUT + base);
        copy16(V + 3 * EMBD, V_LUT + base);
        T[3] = (uint8_t)tok;
    }
    return nxt;
}

HOT_INLINE int step_pos4_precomputed(int tok, float *restrict K, float *restrict V STEP_T_PARAM,
                                     uint32_t *restrict rng) {
    const int base = LUT_BASE(tok, 4);
    const int idx = (((int)T[1] * VOCAB + (int)T[2]) * VOCAB + (int)T[3]) * VOCAB + tok;
    int nxt = sample_cumulative_pos01_shape(CUM_POS4_LUT + idx * LOGIT_STRIDE, SUM_POS4_LUT[idx], rng);
    if (nxt != BOS) {
        copy16(K + 4 * EMBD, K_LUT + base);
        copy16(V + 4 * EMBD, V_LUT + base);
        T[4] = (uint8_t)tok;
    }
    return nxt;
}

HOT_INLINE int step_pos5_precomputed(int tok, float *restrict K, float *restrict V STEP_T_PARAM,
                                     uint32_t *restrict rng) {
    const int base = LUT_BASE(tok, 5);
    const size_t idx = ((((size_t)T[1] * VOCAB + (size_t)T[2]) * VOCAB + (size_t)T[3]) * VOCAB + (size_t)T[4]) * VOCAB + (size_t)tok;
    int nxt = sample_cumulative_pos01_shape(CUM_POS5_LUT + idx * LOGIT_STRIDE, SUM_POS5_LUT[idx], rng);
    if (nxt != BOS) {
        copy16(K + 5 * EMBD, K_LUT + base);
        copy16(V + 5 * EMBD, V_LUT + base);
        T[5] = (uint8_t)tok;
    }
    return nxt;
}


__attribute__((noinline, cold, unused))
static int step_ref(int tok, int pos, float *K, float *V, uint32_t *rng) {
    float xr[EMBD] __attribute__((aligned(64)));
    float x[EMBD] __attribute__((aligned(64)));
    float head_out[EMBD] __attribute__((aligned(64)));
    float logits[LIVE_LOGIT_STRIDE] __attribute__((aligned(64)));

    int base = LUT_BASE(tok, pos);
    const float *q = Q_LUT + base;
    const float *xr0 = XR_LUT + base;
    copy16(K + pos * EMBD, K_LUT + base);
    copy16(V + pos * EMBD, V_LUT + base);

    const float scale = 1.0f / 2.0f;
    int t_n = pos + 1;
    for (int hi = 0; hi < HEAD; hi++) {
        const float *qh = q + hi * HD;
        float32x4_t qv = vld1q_f32(qh);
        float al[BLOCK] __attribute__((aligned(64)));
        float maxl = -1e30f;
        for (int t = 0; t < t_n; t++) {
            const float *kh = K + t * EMBD + hi * HD;
            float dot = vaddvq_f32(vmulq_f32(qv, vld1q_f32(kh)));
            al[t] = dot * scale;
            if (al[t] > maxl) maxl = al[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < t_n; t++) {
            al[t] = expf(al[t] - maxl);
            sum += al[t];
        }
        float inv = 1.0f / sum;
        float32x4_t out = vdupq_n_f32(0.0f);
        for (int t = 0; t < t_n; t++) {
            float w = al[t] * inv;
            const float *vh = V + t * EMBD + hi * HD;
            out = vfmaq_n_f32(out, vld1q_f32(vh), w);
        }
        vst1q_f32(head_out + hi * HD, out);
    }

    wo_project_add_ref(head_out, xr0, x);

    rmsnorm_save(x, xr);

    mlp_fused_add_ref(x, xr, x);

    float maxl = lm_logits_ref(x, logits);
    float sum = 0.0f;
    for (int i = 0; i < VOCAB; i++) {
        logits[i] = expf(logits[i] - maxl);
        sum += logits[i];
    }

    return sample_weights(logits, sum, rng);
}

HOT_INLINE int step(int tok, int pos, float *restrict K, float *restrict V STEP_T_PARAM,
                    StepScratch *restrict scratch, uint32_t *restrict rng) {
    float *x = scratch->x;
    float *logits = scratch->logits;
    const float *softmax_logits = logits;
    const float *pre_logits = NULL;
    float pre_maxl = 0.0f;
    int pre_max_idx = 0;
    const float *pre_cum = NULL;
    float pre_sum = 0.0f;


    int base;
    const float *cur_v;
    const float *xr0;
    if (pos == 0) {
        base = LUT_BASE(BOS, 0);
        cur_v = V_LUT + base;
        xr0 = XR_LUT + base;
    } else {
        base = LUT_BASE(tok, pos);
        cur_v = V_LUT + base;
        xr0 = XR_LUT + base;
    }

    int have_post_mlp = 0;
    if (pos == 0) {
        have_post_mlp = 1;
        pre_logits = LOGITS_POS0_LUT + BOS * LOGIT_STRIDE;
        pre_maxl = MAXL_POS0_LUT[BOS];
        pre_max_idx = MAXI_POS0_LUT[BOS];
        pre_cum = CUM_POS0_LUT + BOS * LOGIT_STRIDE;
        pre_sum = SUM_POS0_LUT[BOS];
    }

    if (pos == 1) {
        have_post_mlp = 1;
        pre_logits = LOGITS_POS1_LUT + tok * LOGIT_STRIDE;
        pre_maxl = MAXL_POS1_LUT[tok];
        pre_max_idx = MAXI_POS1_LUT[tok];
        pre_cum = CUM_POS1_LUT + tok * LOGIT_STRIDE;
        pre_sum = SUM_POS1_LUT[tok];
    } else
    if (pos == 1) {
        const float *q = Q_LUT + base;
        int qk_base = (base >> 4) * HEAD;
        float32x4_t hv0, hv1, hv2, hv3;
        for (int hi = 0; hi < HEAD; hi++) {
            const float *qh = q + hi * HD;
            float32x4_t qv = vld1q_f32(qh);
            const float *kh0 = K + hi * HD;
            float al0 = vaddvq_f32(vmulq_f32(qv, vld1q_f32(kh0))) * 0.5f;
            float al1 = QK_SELF_LUT[qk_base + hi];
            float maxl = (al0 > al1) ? al0 : al1;
            al0 = (al0 == maxl) ? 1.0f : strict_expf(al0 - maxl);
            al1 = (al1 == maxl) ? 1.0f : strict_expf(al1 - maxl);
            float inv = 1.0f / (al0 + al1);
            float32x4_t out = vdupq_n_f32(0.0f);
            out = vfmaq_n_f32(out, vld1q_f32(V + hi * HD), al0 * inv);
            out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al1 * inv);
            if (hi == 0) hv0 = out;
            else if (hi == 1) hv1 = out;
            else if (hi == 2) hv2 = out;
            else hv3 = out;
        }
        wo_project_add_vecs(hv0, hv1, hv2, hv3, xr0, x);
    }
    else if (pos == 2) {
        int qtp = base / EMBD;
        const float *restrict qk_pair_qbase = QK_PAIR_LUT + (size_t)qtp * (VOCAB * BLOCK * HEAD);
        int qk_base = (base >> 4) * HEAD;
        float32x4_t hv0, hv1, hv2, hv3;
        for (int hi = 0; hi < HEAD; hi++) {
            const float *restrict qk_pair_head = qk_pair_qbase + hi;
            float al0 = qk_pair_head[STEP_TP(T, 0) * HEAD];
            float al1 = qk_pair_head[STEP_TP(T, 1) * HEAD];
            float al2 = QK_SELF_LUT[qk_base + hi];
            float maxl = al0;
            if (al1 > maxl) maxl = al1;
            if (al2 > maxl) maxl = al2;
            al0 = (al0 == maxl) ? 1.0f : strict_expf(al0 - maxl);
            al1 = (al1 == maxl) ? 1.0f : strict_expf(al1 - maxl);
            al2 = (al2 == maxl) ? 1.0f : strict_expf(al2 - maxl);
            float inv = 1.0f / ((al0 + al1) + al2);
            float32x4_t out = vdupq_n_f32(0.0f);
            out = vfmaq_n_f32(out, vld1q_f32(STEP_V_PTR(T, V, 0, hi)), al0 * inv);
            out = vfmaq_n_f32(out, vld1q_f32(STEP_V_PTR(T, V, 1, hi)), al1 * inv);
            out = vfmaq_n_f32(out, vld1q_f32(cur_v + hi * HD), al2 * inv);
            if (hi == 0) hv0 = out;
            else if (hi == 1) hv1 = out;
            else if (hi == 2) hv2 = out;
            else hv3 = out;
        }
        wo_project_add_vecs(hv0, hv1, hv2, hv3, xr0, x);
    }
    else if (pos != 0) {
        attention_generic(pos, base, cur_v, K, V
                          , T
                          , xr0, x, scratch->attn);
    }

    int max_idx;
    float maxl;
    if (pre_logits) {
        softmax_logits = pre_logits;
        maxl = pre_maxl;
        max_idx = pre_max_idx;
    } else
    if (have_post_mlp) {
        maxl = lm_logits(x, logits, &max_idx);
    } else {
    float32x4_t r0, r1, r2, r3;
    float32x4_t n0, n1, n2, n3;
    RMSNORM_VECS_LOCAL(x, r0, r1, r2, r3, n0, n1, n2, n3);

    mlp_fused_add_vecs(n0, n1, n2, n3, r0, r1, r2, r3, x);

    maxl = lm_logits(x, logits, &max_idx);

    }
    int nxt;
    if (pre_cum) {
        nxt = sample_cumulative(pre_cum, pre_sum, rng);
        goto sample_done;
    }
    float sum = build_cumulative_live(softmax_logits, maxl, max_idx, logits);
    nxt = sample_cumulative_neon(logits, sum, rng);
sample_done:
    if (nxt != BOS && pos != BLOCK - 1) {
        copy16(K + pos * EMBD, K_LUT + base);
        copy16(V + pos * EMBD, cur_v);
        T[pos] = (uint8_t)((pos == 0) ? BOS : tok);
    }
    return nxt;
}


static void apply_benchmark_thread_policy(void) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(W, sizeof(float), TOTAL, f) != TOTAL) {
        fprintf(stderr, "short read\n"); exit(1);
    }
    fclose(f);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define RESET_UNLIKELY(nxt_) __builtin_expect((nxt_) == BOS, 0)
#define RESET_LIKELY(nxt_) __builtin_expect((nxt_) == BOS, 1)

__attribute__((noinline, hot))
static void run_tokens_unrolled(long N, int *tokp, int *posp,
                                float *restrict K, float *restrict V STEP_T_PARAM,
                                StepScratch *restrict scratch, uint32_t *rngp) {
    long i = 0;
    int tok = *tokp;
    int pos = *posp;
    uint32_t rng = *rngp;
    int nxt;

    while (i < N) {
        switch (pos) {
            case 0: goto pos0;
            case 1: goto pos1;
            case 2: goto pos2;
            case 3: goto pos3;
            case 4: goto pos4;
            case 5: goto pos5;
            case 6: goto pos6;
            case 7: goto pos7;
            case 8: goto pos8;
            case 9: goto pos9;
            case 10: goto pos10;
            case 11: goto pos11;
            case 12: goto pos12;
            case 13: goto pos13;
            case 14: goto pos14;
            default: goto pos15;
        }

pos0:
        if (i >= N) break;
        nxt = step_pos0_precomputed(K, V STEP_T_ARG, &rng);
        i++;
        if (RESET_UNLIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 1;
pos1:
        if (i >= N) break;
        nxt = step_pos1_precomputed(tok, K, V STEP_T_ARG, &rng);
        i++;
        if (RESET_UNLIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 2;
pos2:
        if (i >= N) break;
        nxt = step_pos2_precomputed(tok, K, V STEP_T_ARG, &rng);
        i++;
        if (RESET_UNLIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 3;
pos3:
        if (i >= N) break;
        nxt = step_pos3_precomputed(tok, K, V STEP_T_ARG, &rng);
        i++;
        if (RESET_UNLIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 4;
pos4:
        if (i >= N) break;
        nxt = step_pos4_precomputed(tok, K, V STEP_T_ARG, &rng);
        i++;
        if (RESET_UNLIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 5;
pos5:
        if (i >= N) break;
        nxt = step_pos5_precomputed(tok, K, V STEP_T_ARG, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 6;
pos6:
        if (i >= N) break;
        nxt = step(tok, 6, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 7;
pos7:
        if (i >= N) break;
        nxt = step(tok, 7, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 8;
pos8:
        if (i >= N) break;
        nxt = step(tok, 8, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 9;
pos9:
        if (i >= N) break;
        nxt = step(tok, 9, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 10;
pos10:
        if (i >= N) break;
        nxt = step(tok, 10, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 11;
pos11:
        if (i >= N) break;
        nxt = step(tok, 11, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 12;
pos12:
        if (i >= N) break;
        nxt = step(tok, 12, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 13;
pos13:
        if (i >= N) break;
        nxt = step(tok, 13, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 14;
pos14:
        if (i >= N) break;
        nxt = step(tok, 14, K, V STEP_T_ARG, scratch, &rng);
        i++;
        if (RESET_LIKELY(nxt)) { tok = BOS; pos = 0; continue; }
        tok = nxt; pos = 15;
pos15:
        if (i >= N) break;
        (void)step(tok, 15, K, V STEP_T_ARG, scratch, &rng);
        i++;
        tok = BOS; pos = 0;
    }

    *tokp = tok;
    *posp = pos;
    *rngp = rng;
}
#undef RESET_LIKELY
#undef RESET_UNLIKELY

__attribute__((noinline, hot))
static void bench_single(long N, long WUP) {
    apply_benchmark_thread_policy();
    float K[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
    float V[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
    STEP_T_TYPE T[BLOCK] __attribute__((aligned(64))) = {0};
    StepScratch scratch;
    uint32_t rng = 42;
    int tok = BOS, pos = 0;

    for (long i = 0; i < WUP; i++) {
        int nxt = step(tok, pos, K, V STEP_T_ARG, &scratch, &rng);
        if (nxt == BOS || pos == BLOCK - 1) { tok = BOS; pos = 0; }
        else { tok = nxt; pos++; }
    }
    double t0 = now_sec();
    run_tokens_unrolled(N, &tok, &pos, K, V STEP_T_ARG, &scratch, &rng);
    double t1 = now_sec();
    double rate = N / (t1 - t0);
    printf("  %-27s %14.0f tok/sec\n", BENCH_LABEL, rate);
}


int main(int argc, char **argv) {
    long N = (argc > 1) ? atol(argv[1]) : 5000000;
    long WUP = (argc > 2) ? atol(argv[2]) : 100000;
    int check_mode = (argc > 1 && strcmp(argv[1], "--check") == 0);
    int driver_check_mode = (argc > 1 && strcmp(argv[1], "--driver-check") == 0);

    load_weights("assets/weights_fp32.bin");
    precompute_w2_transpose();
    precompute_front_half();
    precompute_pos01_mlp_luts();
    precompute_pos2_cum_luts();
    precompute_pos3_cum_luts();
    precompute_pos4_cum_luts();
    precompute_pos5_cum_luts();

    if (check_mode) {
        long C = (argc > 2) ? atol(argv[2]) : 500000;
        float Kr[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        float Vr[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        float Ko[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        float Vo[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        STEP_T_TYPE T[BLOCK] __attribute__((aligned(64))) = {0};
        StepScratch scratch_opt;
        uint32_t rng_ref = 42;
        uint32_t rng_opt = 42;
        int tok_ref = BOS, pos_ref = 0;
        int tok_opt = BOS, pos_opt = 0;

        for (long i = 0; i < C; i++) {
            int ref = step_ref(tok_ref, pos_ref, Kr, Vr, &rng_ref);
            int opt = step(tok_opt, pos_opt, Ko, Vo STEP_T_ARG, &scratch_opt, &rng_opt);
            if (ref != opt || rng_ref != rng_opt || tok_ref != tok_opt || pos_ref != pos_opt) {
                printf("strict check mismatch at %ld: ref=%d opt=%d rng_ref=%u rng_opt=%u tok_ref=%d tok_opt=%d pos_ref=%d pos_opt=%d\n",
                       i, ref, opt, rng_ref, rng_opt, tok_ref, tok_opt, pos_ref, pos_opt);
                return 1;
            }

            if (ref != BOS && pos_ref != BLOCK - 1) {
                size_t active = (size_t)(pos_ref + 1) * EMBD * sizeof(float);
                if (memcmp(Kr, Ko, active) != 0 || memcmp(Vr, Vo, active) != 0) {
                    printf("strict check KV mismatch at %ld active_pos=%d\n", i, pos_ref + 1);
                    return 1;
                }
            }

            if (ref == BOS || pos_ref == BLOCK - 1) { tok_ref = BOS; pos_ref = 0; }
            else { tok_ref = ref; pos_ref++; }

            if (opt == BOS || pos_opt == BLOCK - 1) { tok_opt = BOS; pos_opt = 0; }
            else { tok_opt = opt; pos_opt++; }
        }
        printf("strict check ok: %ld tokens\n", C);
        return 0;
    }

    if (driver_check_mode) {
        long C = (argc > 2) ? atol(argv[2]) : 5000000;
        float Kn[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        float Vn[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        float Ku[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        float Vu[BLOCK * EMBD] __attribute__((aligned(64))) = {0};
        STEP_T_TYPE Tn[BLOCK] __attribute__((aligned(64))) = {0};
        STEP_T_TYPE Tu[BLOCK] __attribute__((aligned(64))) = {0};
        StepScratch scratch_n;
        StepScratch scratch_u;
        uint32_t rng_n = 42;
        uint32_t rng_u = 42;
        int tok_n = BOS, pos_n = 0;
        int tok_u = BOS, pos_u = 0;
        for (long i = 0; i < C; i++) {
            int nxt = step(tok_n, pos_n, Kn, Vn STEP_T_ARG_NAMED(Tn), &scratch_n, &rng_n);
            if (nxt == BOS || pos_n == BLOCK - 1) { tok_n = BOS; pos_n = 0; }
            else { tok_n = nxt; pos_n++; }
        }
        run_tokens_unrolled(C, &tok_u, &pos_u, Ku, Vu STEP_T_ARG_NAMED(Tu), &scratch_u, &rng_u);
        size_t active = (size_t)pos_n * EMBD * sizeof(float);
        if (tok_n != tok_u || pos_n != pos_u || rng_n != rng_u ||
            memcmp(Kn, Ku, active) != 0 || memcmp(Vn, Vu, active) != 0) {
            printf("driver check mismatch: tok %d/%d pos %d/%d rng %u/%u active=%zu\n",
                   tok_n, tok_u, pos_n, pos_u, rng_n, rng_u, active);
            return 1;
        }
        printf("driver check ok: %ld tokens\n", C);
        return 0;
    }

    bench_single(N, WUP);
    return 0;
}
