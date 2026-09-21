"""Measure NAX matmul2d instruction rates for EVERY supported type combination.

Sweeps all rows of the matmul2d support table (MPPTensorOpsMatMul2d.h header
comment == MSL 4.1 spec Table 7.3) on this machine, normalized to
int8 x int8 -> int32 = 1.0.

Two harness variants, best result reported per combo:
- "tg": A/B staged in threadgroup memory as tensor_inline views (works for all
  combos incl. packed 4b/2b/fp8/fp4 formats, which take a uchar* handle).
  Reaches ~91% of the register-resident rate for int8, ~100% for fp16.
- "coop": register-resident cooperative input tensors (mma_rate.py harness);
  only compiles for matched-operand combos.

Writes mma_rate_all.json next to this file. Run: .venv/bin/python research/int8-nax/mma_rate_all.py
"""

import json
import os
import time

import mlx.core as mx

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;

template <typename T> struct is_packed_fmt { static constant constexpr bool value = false; };
#define PACKED(T) template <> struct is_packed_fmt<T> { static constant constexpr bool value = true; };
PACKED(int4b_format) PACKED(uint4b_format) PACKED(int2b_format) PACKED(uint2b_format)
PACKED(metal_fp8_e4m3_format) PACKED(metal_fp8_e5m2_format) PACKED(metal_fp4_e2m1_format)

template <typename T>
METAL_FUNC auto make_tg_tensor(threadgroup uint32_t* p, dextents<int32_t,2> e) {
    if constexpr (is_packed_fmt<T>::value)
        return tensor<threadgroup T, dextents<int32_t,2>, tensor_inline>((threadgroup uchar*)p, e);
    else
        return tensor<threadgroup T, dextents<int32_t,2>, tensor_inline>((threadgroup T*)p, e);
}

template <typename AT, typename BT, typename CT, int TM, int TN, int TK, int ITERS, bool RELAX>
METAL_FUNC int run_tg_loop(threadgroup uint32_t* bufA, threadgroup uint32_t* bufB, uint tid) {
    for (int i = int(tid); i < TM*TK; i += 128) bufA[i] = 0x11111111u;
    for (int i = int(tid); i < TK*TN; i += 128) bufB[i] = 0x11111111u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, TK, false, false, RELAX,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;
    auto A = make_tg_tensor<AT>(bufA, dextents<int32_t,2>(TK, TM));
    auto B = make_tg_tensor<BT>(bufB, dextents<int32_t,2>(TN, TK));
    auto c = op.template get_destination_cooperative_tensor<decltype(A), decltype(B), CT>();
    for (int i = 0; i < int(c.get_capacity()); i++) c[i] = CT(0);
    for (int it = 0; it < ITERS; it++) op.run(A, B, c);
    int acc = 0;
    for (int i = 0; i < int(c.get_capacity()); i++)
        if (c.is_valid_element(i)) acc += int(c[i]);
    return acc;
}

template <typename AT, typename BT, typename CT, int TM, int TN, int TK, int ITERS, bool RELAX>
METAL_FUNC int run_coop_loop(uint seed) {
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, TK, false, false, RELAX,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;
    auto a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), CT>();
    for (int i = 0; i < int(a.get_capacity()); i++) a[i] = AT(seed & 3);
    for (int i = 0; i < int(b.get_capacity()); i++) b[i] = BT((seed >> 2) & 3);
    for (int i = 0; i < int(c.get_capacity()); i++) c[i] = CT(0);
    for (int it = 0; it < ITERS; it++) op.run(a, b, c);
    int acc = 0;
    for (int i = 0; i < int(c.get_capacity()); i++) acc += int(c[i]);
    return acc;
}
"""

ITERS, TGS, TPT = 20000, 2048, 128
TILES = [(16, 32, 32), (32, 32, 32), (32, 32, 64), (64, 32, 32)]

_counter = [0]


def _run_kernel(variant, AT, BT, CT, TM, TN, TK, relax):
    relax_s = "true" if relax else "false"
    if variant == "tg":
        src = f"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup uint32_t bufA[{TM}*{TK}];
    threadgroup uint32_t bufB[{TK}*{TN}];
    int r = run_tg_loop<{AT}, {BT}, {CT}, {TM}, {TN}, {TK}, {ITERS}, {relax_s}>(bufA, bufB, tid);
    if (r == -12345) out[0] = r;
"""
    else:
        src = f"""
    uint gid = thread_position_in_grid.x;
    int r = run_coop_loop<{AT}, {BT}, {CT}, {TM}, {TN}, {TK}, {ITERS}, {relax_s}>(gid);
    if (r == -12345) out[0] = r;
"""
    _counter[0] += 1
    k = mx.fast.metal_kernel(
        name=f"mma_{variant}_{_counter[0]}",
        input_names=["dummy"], output_names=["out"],
        header=HEADER, source=src)
    f = lambda: k(inputs=[mx.zeros(1, dtype=mx.int32)],
                  grid=(TGS * TPT, 1, 1), threadgroup=(TPT, 1, 1),
                  output_shapes=[(1,)], output_dtypes=[mx.int32])
    mx.eval(f()); mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        mx.eval(f())
    mx.synchronize()
    t = (time.perf_counter() - t0) / 3
    simds = TGS * (TPT // 32)
    return 2.0 * TM * TN * TK * ITERS * simds / t / 1e12


def bench_combo(AT, BT, CT, relax=True):
    """Returns (best_tops, best_cfg, per_cfg_dict). Failed configs recorded as None."""
    results = {}
    best, best_cfg = 0.0, None
    for TM, TN, TK in TILES:
        for variant in ("tg", "coop") if AT == BT else ("tg",):
            key = f"{variant}_{TM}x{TN}x{TK}"
            try:
                tops = _run_kernel(variant, AT, BT, CT, TM, TN, TK, relax)
                results[key] = round(tops, 1)
                if tops > best:
                    best, best_cfg = tops, key
            except Exception as e:
                lines = [l for l in str(e).splitlines() if "error:" in l]
                results[key] = "FAIL: " + (lines[0][-120:] if lines else str(e)[:120])
    return best, best_cfg, results


# Every row of the header/spec support table (A, B, C[, tag]).
FP8_43 = "metal_fp8_e4m3_format"
FP8_52 = "metal_fp8_e5m2_format"
FP4 = "metal_fp4_e2m1_format"

COMBOS = [
    # --- Metal 4 base: half/float/int8 families ---
    ("half", "half", "half"),
    ("half", "int8_t", "half"),
    ("half", "uint8_t", "half"),
    ("int8_t", "half", "half"),
    ("uint8_t", "half", "half"),
    ("half", "half", "float"),
    ("half", "float", "float"),
    ("half", "int8_t", "float"),
    ("half", "uint8_t", "float"),
    ("float", "half", "float"),
    ("float", "float", "float"),          # relaxed (tf32-style) by default flag below
    ("float", "float", "float", "strict"),
    ("float", "int8_t", "float"),
    ("float", "uint8_t", "float"),
    ("int8_t", "half", "float"),
    ("uint8_t", "half", "float"),
    ("int8_t", "float", "float"),
    ("uint8_t", "float", "float"),
    ("int8_t", "int8_t", "int32_t"),
    ("uint8_t", "uint8_t", "int32_t"),
    # --- bfloat family (OS 26.1) ---
    ("bfloat", "bfloat", "bfloat"),
    ("bfloat", "bfloat", "float"),
    ("bfloat", "float", "float"),
    ("bfloat", "int8_t", "bfloat"),
    ("bfloat", "int8_t", "float"),
    ("float", "bfloat", "float"),
    ("int8_t", "bfloat", "bfloat"),
    ("int8_t", "bfloat", "float"),
    ("bfloat", "half", "bfloat"),
    ("bfloat", "half", "half"),
    ("bfloat", "half", "float"),
    ("half", "bfloat", "bfloat"),
    ("half", "bfloat", "half"),
    ("half", "bfloat", "float"),
    ("bfloat", "uint8_t", "bfloat"),
    ("bfloat", "uint8_t", "float"),
    ("uint8_t", "bfloat", "bfloat"),
    ("uint8_t", "bfloat", "float"),
    # --- 4-bit weight formats (OS 26.4) ---
    ("half", "int4b_format", "half"),
    ("half", "int4b_format", "float"),
    ("half", "uint4b_format", "half"),
    ("half", "uint4b_format", "float"),
    ("int8_t", "int4b_format", "int32_t"),
    ("uint8_t", "uint4b_format", "int32_t"),
    ("bfloat", "int4b_format", "bfloat"),
    ("bfloat", "uint4b_format", "bfloat"),
    ("bfloat", "int4b_format", "float"),
    ("bfloat", "uint4b_format", "float"),
    # --- 2-bit weight formats (Metal 4.1) ---
    ("int8_t", "int2b_format", "int32_t"),
    ("uint8_t", "uint2b_format", "int32_t"),
    ("half", "int2b_format", "half"),
    ("half", "int2b_format", "float"),
    ("half", "uint2b_format", "half"),
    ("half", "uint2b_format", "float"),
    ("bfloat", "int2b_format", "bfloat"),
    ("bfloat", "uint2b_format", "bfloat"),
    ("bfloat", "int2b_format", "float"),
    ("bfloat", "uint2b_format", "float"),
    # --- fp8 / fp4 (Metal 4.1) ---
    ("half", FP4, "half"),
    ("half", FP4, "float"),
    ("half", FP8_43, "half"),
    ("half", FP8_43, "float"),
    ("half", FP8_52, "half"),
    ("half", FP8_52, "float"),
    (FP4, FP4, "half"),
    (FP4, FP4, "float"),
    (FP8_43, FP8_43, "half"),
    (FP8_43, FP8_43, "float"),
    (FP8_52, FP8_52, "half"),
    (FP8_52, FP8_52, "float"),
]


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mma_rate_all.json")
    rows = []
    for combo in COMBOS:
        AT, BT, CT = combo[:3]
        strict = len(combo) > 3 and combo[3] == "strict"
        label = f"{AT} x {BT} -> {CT}" + (" (strict)" if strict else "")
        t0 = time.perf_counter()
        best, best_cfg, per_cfg = bench_combo(AT, BT, CT, relax=not strict)
        rows.append({"a": AT, "b": BT, "c": CT, "strict": strict,
                     "best_tops": round(best, 1), "best_cfg": best_cfg,
                     "configs": per_cfg})
        print(f"{label:55s} {best:7.1f} TOPS  [{best_cfg}]  ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        with open(out_path, "w") as fh:   # checkpoint after every combo
            json.dump(rows, fh, indent=1)

    base = next(r["best_tops"] for r in rows
                if r["a"] == "int8_t" and r["b"] == "int8_t")
    print(f"\nbaseline int8 x int8 -> int32: {base} TOPS")
    for r in sorted(rows, key=lambda r: -r["best_tops"]):
        rel = r["best_tops"] / base if base else 0
        tag = " (strict)" if r["strict"] else ""
        print(f"{r['a']:24s} {r['b']:24s} {r['c']:8s}{tag:9s} "
              f"{r['best_tops']:7.1f} TOPS  rel {rel:5.2f}")


if __name__ == "__main__":
    main()
