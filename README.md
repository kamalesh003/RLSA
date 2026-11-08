# RLSA

## What R-LSA is for

R-LSA (Reverse Latent Sparse Attention) is a way to make the global attention layers of a long-context decoder-only model cheaper. Full attention costs O(T²) compute, and decoding has to read every cached key for each new token. R-LSA lets each query look at a small, content-selected subset of past tokens instead.

In your code, a global layer's query tile sees four things:

1. **Sink tokens:** 4 learned, always-visible K/V slots per head.
2. **Local window:** the previous 256 tokens plus the 16-token tile, with causal masking.
3. **Retrieved remote keys:** at most 256 old tokens, chosen through the index.
4. **Anchor summary tokens:** up to 64, each the mass-weighted average K/V of an anchor over all closed blocks. Keys already retrieved are subtracted, so nothing is counted twice.

The "reverse" part is that the routing work happens at write time. Each key is assigned once to its top 2 anchors within its 256-token block, and each (block, anchor) bucket holds at most 16 keys. At read time a query tile picks its top 6 anchors, then the 32 best buckets across eligible blocks, then expands to member keys and dedupes. It never scores individual keys.

The attention step is capped at about 596 keys per tile regardless of context length. The read step scores centroids in every eligible block, and the index build touches every key, so total cost grows roughly linearly with T, not as T².

In the hybrid model, six of the eight layers are sliding-window attention with RoPE. Layers 4 and 8 are global NoPE layers that use R-LSA for long-range retrieval.

## Where it could be used

These are design targets. This repo doesn't test any of them.

- **Long-document and long-code models:** books, legal and medical records, repository-level code, long chat histories.
- **Long-context decoding:** reading about 600 keys per step instead of the full cache is the biggest potential saving. The repo has no incremental decode path yet, and all K/V must still be stored, so the saving would be in reads, not memory.
- **Memory-style and agent models:** retrieval over a long past inside the model, instead of an external RAG system.
- **Other very long sequences:** genomics, audio, logs, time series, long-video tokens.

It isn't useful at short context. At 1K to 4K tokens, a fused dense or flash attention kernel is hard to beat.

## What your own run shows

**Speed (Phase 5, T=1024, forward only):**

| Mode | Latency | vs dense |
|---|---|---|
| dense | 38.35 ms | 1.00× |
| swa | 38.66 ms | 1.01× |
| rlsa | 70.80 ms | 1.85× slower |
| rlsa_no_summaries | 69.71 ms | 1.82× slower |
| flat_lsa | 75.22 ms | 1.96× slower |

At 1K there is nothing to save, because R-LSA attends to about as many keys as dense and adds indexing and retrieval work. Summaries cost only about 1 ms. Peak VRAM is identical across modes (0.136 GiB), so it says nothing about attention memory.

**Quality:** all five modes land between 6.68883 and 6.68937 test loss (perplexity about 803.5), a spread of about 0.0005. That is noise. Even dense and sliding-window are identical, so the model isn't using anything beyond 256 tokens back yet.

The model is barely trained. After 400 steps it sits at roughly word-frequency level. The stage 2 improvement (validation loss 6.86 to 6.54) can't be credited to R-LSA, because it had 200 more steps at a fresh learning rate.

**Retrieval is barely used.** Only blocks that end at least 257 tokens behind the query are searchable, so at T=1024 only 31 of 64 tiles can retrieve anything. The logged mean remote keys per tile is 23.3, against a budget of 256.

**Router health is still skewed.** The last stage 2 log shows:

- **Overflow:** 22.0%, down from 45.5% at step 50.
- **Write load per bucket:** 8.0 ± 10.1, with a max of 143.
- **Anchor use:** the routing queries use 26 of 64 anchors, while the EMA statistics show 51 of 64 active. The write side is unbalanced even though capacity is double the writes per block.

## Next steps

1. Train much longer, until perplexity is well below 200.
2. Add a dense-continuation control for stage 2.
3. Benchmark at 2048 and 4096 to see where the latency curves cross.
4. Add a synthetic recall test, with a fact planted 500 to 3,000 tokens back.
5. Strengthen load balancing (`balance_step`) or raise `bucket_capacity`.
6. Run `phase6_profile.py` to see where the extra 32 ms goes.



# R-LSA: reverse write and read

The idea is an inverted index over keys. In a normal sparse retriever, each query scores many keys at read time. Here the keys do the routing work up front, by publishing themselves into anchor-labelled buckets. The query then only has to find the right buckets. The code is in `index.py` (`build_reverse_index`, `retrieve_remote_keys_batched`) and `attention.py`.

The numbers below come from your target config: block 256, 64 anchors, write top-r 2, bucket capacity 16, query top-A 6, bucket top-P 32, remote budget 256, local window 256, 4 sinks, tile 16.

## Anchors

Anchors are 64 unit vectors in a 32-dim index space. They are not trained by gradient. They act as query prototypes, updated by an EMA spherical k-means step (`ema_update_anchors`) from the layer's query features. Empty anchors are reseeded from the least-represented queries. This is why Stage 1 (dense warm-up) still runs the EMA: the anchors are already meaningful when R-LSA switches on.

## Write: keys post themselves to anchors

This happens once per forward pass, per global layer, in `build_reverse_index`, and it is fully vectorized.

1. **Project.** Each key goes through `k_index` (head_dim → 32) and is normalized. Only complete 256-token blocks are indexed.
2. **Assign top-r.** Each key scores its cosine similarity to all 64 anchors and writes to its top 2.
3. **Load balance.** MoE-style, with no auxiliary loss. Before the final top-r, a bias is added to the scores and nudged for 8 vectorized rounds. Anchors that are over-subscribed in a block get pushed down, and under-subscribed ones get pushed up. A persistent `load_balance_bias` buffer also drifts across training steps. Target load is 256 × 2 / 64 = 8 postings per (block, anchor).
4. **Bounded buckets.** Postings are grouped by (block, anchor), and each bucket holds at most 16 keys, which is 2× the target load. If a bucket overflows, it keeps the 16 keys with the highest raw similarity. The balance bias is not used for this ranking. Overflow is tracked as `bucket_overflow_rate`.
5. **Per-bucket metadata.** Each bucket stores:
   - `members`: absolute token positions.
   - `centroid`: the normalized mean of its members' index features.
   - Mass-weighted K and V sums. A key contributes weight 1/r, so its mass is split across its two postings.
6. **Prefix sums.** The K, V and mass sums are cumulative over blocks. Later, "summarize everything up to block n" is a single lookup.
7. **Retained-write record.** `token_retained_anchors` records which of each token's r writes survived capacity clipping. The read side needs this to avoid double-counting.

## Read: two-stage, per query tile

Queries are grouped into tiles of 16. Each tile gets one routing query, the `q_index` feature of the tile's first token (the mean over heads). Up to 32 tiles are processed in one batched call.

1. **Causal cutoff.** `cutoff = tile_start − local_window − 1`. Only blocks fully closed at or before the cutoff are eligible. Everything newer is handled by the local window, so the two paths never overlap.
2. **Stage 1, anchor selection.** Score the routing query against all 64 anchors and keep the top 6.
3. **Stage 2, bucket selection.** For each eligible block and each of the 6 anchors, score the bucket as `cos(centroid, q) + 0.1 × anchor_score`. Empty buckets are masked. Take the top 32 buckets across all blocks.
4. **Expand and dedupe.** 32 buckets × 16 members gives up to 512 candidates. A key written to two anchors may appear twice. A stable-sort dedupe keeps the first occurrence in best-first order, and the result is truncated to 256 keys. Each key carries its bucket's score as a routing bias.
5. **Gather.** The real K/V for those 256 positions are fetched.

The `flat_lsa` ablation skips Stage 1 and scans every bucket's centroid directly, with the same 32-bucket cap.

## What the attention actually sees

For each tile, SDPA runs over one concatenated key set:

| Segment | Size | Additive logit bias |
|---|---|---|
| Sink tokens (learned) | 4 | 0 |
| Retrieved remote keys | ≤256 | bucket score (routing bias) |
| Anchor summary tokens | 64 | log(mass) |
| Local window (causal-masked) | 256 + 16 | 0 |

That is about 600 keys per query instead of all T.

**Summary tokens** give coarse coverage of everything not retrieved. For each anchor, the model takes the mass-weighted mean K/V over all closed blocks up to the cutoff, using the prefix sums. It then subtracts the contribution of the keys it just retrieved explicitly (via `token_retained_anchors`), so nothing is counted twice. The `log(mass)` bias makes a summary token behave like roughly "mass-many" keys in the softmax. `rlsa_no_summaries` ablates this.

## Where gradients flow

Top-k selection is not differentiable. The routing bias is what carries gradient: it is built from centroids (derived from `k_index`) and the query feature (from `q_index`), and it is added to the attention logits. This trains both index projections, which `phase2` and `phase3` assert. Anchors and the load-balance bias are updated outside backprop.

## Where it sits in the model

- Only the global layers use R-LSA, which is every 4th layer (layers 4 and 8). They use NoPE.
- The other layers are sliding-window attention with RoPE.
- K/V is multi-query, with a single head shared by all 5 query heads. The index is therefore shared across heads, and the routing query is the head-mean.




**To Run:**

PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> py -m phases.phase1 --device cuda --preserve-downstream
PHASE 1 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase1_pass.json

PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> py -m phases.phase2 --device cuda --preserve-downstream
PHASE 2 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase2_pass.json

PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> py -m phases.phase3 --device cuda --preserve-downstream
PHASE 3 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase3_pass.json

PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> py -m phases.phase4_train --device cuda --amp bf16 --stage1-steps 200 --stage2-steps 200 --seq-len-stage1 512 --seq-len-stage2 1024 --batch-size 1 --grad-accum 2
Device: cuda | model parameters: 11,431,552 | vocab: 8000
Dataset tokens: train=5,000,000 val=200,000 test=200,000
stage1_dense step   50/200 | train 7.7419 | val 7.2919 | ppl 1468.4 | lr 2.61e-04 | tok/s 5823
stage1_dense step  100/200 | train 7.1183 | val 7.0265 | ppl 1126.0 | lr 1.62e-04 | tok/s 5889
stage1_dense step  150/200 | train 6.9250 | val 6.8689 | ppl 961.9 | lr 6.26e-05 | tok/s 6124
stage1_dense step  200/200 | train 6.9008 | val 6.8577 | ppl 951.2 | lr 2.00e-05 | tok/s 6305
stage2_rlsa step   50/200 | train 6.8206 | val 6.7925 | ppl 891.1 | lr 8.87e-05 | tok/s 2520 | remote 17.4 | overflow 0.455 | load 8.0+/-17.7 max 127 | bucket 4.4+/-6.0 max 16 | qA 23/64 std 2.9 max 22 | emaA 53/64 std 44.9 max 339
stage2_rlsa step  100/200 | train 6.7446 | val 6.7694 | ppl 870.8 | lr 6.06e-05 | tok/s 2520 | remote 19.4 | overflow 0.352 | load 8.0+/-14.3 max 96 | bucket 5.2+/-6.1 max 16 | qA 25/64 std 2.3 max 15 | emaA 53/64 std 34.1 max 239
stage2_rlsa step  150/200 | train 6.6931 | val 6.6809 | ppl 797.1 | lr 3.22e-05 | tok/s 2521 | remote 22.1 | overflow 0.263 | load 8.0+/-11.5 max 72 | bucket 5.9+/-5.9 max 16 | qA 26/64 std 1.9 max 10 | emaA 53/64 std 26.2 max 153
stage2_rlsa step  200/200 | train 6.6867 | val 6.5367 | ppl 690.0 | lr 2.00e-05 | tok/s 2535 | remote 23.3 | overflow 0.220 | load 8.0+/-10.1 max 60 | bucket 6.2+/-5.7 max 16 | qA 26/64 std 1.8 max 10 | emaA 51/64 std 24.9 max 143
PHASE 4 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase4_pass.json

PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> py -m phases.phase5_benchmark --device cuda --amp bf16 --split test --seq-len 1024 --batch-size 1 --eval-batches 8
Checkpoint: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\runs\reverse_lsa_wt103_small\last.pt
Loaded stage=stage2_rlsa step=200 | params=11,431,552
Device=cuda | split=test | seq_len=1024 | batches=8
All modes use identical checkpoint weights and identical evaluation windows.
dense              | loss 6.68937 | ppl 803.82 | forward 38.35 ms | tok/s 26702 | VRAM 0.13602304458618164
swa                | loss 6.68926 | ppl 803.73 | forward 38.66 ms | tok/s 26491 | VRAM 0.13602304458618164
flat_lsa           | loss 6.68898 | ppl 803.50 | forward 75.22 ms | tok/s 13613 | VRAM 0.13605260848999023
rlsa               | loss 6.68901 | ppl 803.53 | forward 70.80 ms | tok/s 14464 | VRAM 0.13605260848999023
rlsa_no_summaries  | loss 6.68883 | ppl 803.38 | forward 69.71 ms | tok/s 14689 | VRAM 0.13605260848999023
PHASE 5 BENCHMARK COMPLETE: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase5_benchmark.json
This is a benchmark report, not yet a scientific PASS gate.



PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> py -m phases.phase6_profile --device cuda --amp bf16 --seq-len 1024 --mode rlsa
USDT:2026-11-09 00:22:05 15212:27404 C:\actions-runner\_work\pytorch\pytorch\third_party\kineto\libkineto\src\SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-11-09 00:22:06 15212:27404 C:\actions-runner\_work\pytorch\pytorch\third_party\kineto\libkineto\src\SyncActivityProfilerHandler.cpp:46] profiler_stop
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                               rlsa/reverse_index_build         0.00%       0.000us         0.00%       0.000us       0.000us     309.231ms       162.48%     309.231ms  51.538ms           0 B           0 B           0 B           0 B             6  
                    rlsa/batched_hierarchical_retrieval         0.00%       0.000us         0.00%       0.000us       0.000us     214.803ms       112.87%     214.803ms  17.900ms           0 B           0 B           0 B           0 B            12  
                          rlsa/index_feature_projection         0.00%       0.000us         0.00%       0.000us       0.000us      79.397ms        41.72%      79.397ms  13.233ms           0 B           0 B           0 B           0 B             6  
                                               aten::mm        12.97%     194.308ms        20.80%     311.709ms       1.463ms      46.353ms        24.36%      46.353ms 217.619us           0 B           0 B     193.47 MB     193.47 MB           213  
                        rlsa/summary_prefix_and_overlap         0.00%       0.000us         0.00%       0.000us       0.000us      46.344ms        24.35%      46.344ms   3.862ms           0 B           0 B           0 B           0 B            12  
                     aten::_efficient_attention_forward         0.16%       2.379ms         0.53%       7.923ms     264.100us      24.720ms        12.99%      25.013ms 833.750us           0 B           0 B      15.03 MB           0 B            30  
_Z40fmha_cutlassF_bf16_aligned_64x64_rf_sm80N22PyTor...         0.00%       0.000us         0.00%       0.000us       0.000us      24.720ms        12.99%      24.720ms 823.987us           0 B           0 B           0 B           0 B            30  
                                            aten::copy_         2.44%      36.603ms         4.60%      68.905ms      63.979us      19.509ms        10.25%      19.509ms  18.114us           0 B           0 B           0 B           0 B          1077  
                                              aten::cat         0.87%      12.977ms         1.55%      23.278ms     172.430us      15.843ms         8.32%      15.843ms 117.358us           0 B           0 B     322.73 MB     322.73 MB           135  
_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22g...         0.00%       0.000us         0.00%       0.000us       0.000us      14.574ms         7.66%      14.574ms  69.402us           0 B           0 B           0 B           0 B           210  
                                             aten::topk         0.78%      11.668ms         3.77%      56.496ms     672.573us      13.591ms         7.14%      13.591ms 161.798us           0 B           0 B       1.63 MB       1.63 MB            84  
                                    rlsa/sdpa_attention         0.00%       0.000us         0.00%       0.000us       0.000us      12.905ms         6.78%      12.905ms   1.075ms           0 B           0 B           0 B           0 B            12  
ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us      11.923ms         6.26%      11.923ms 248.400us           0 B           0 B           0 B           0 B            48  
ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      10.582ms         5.56%      10.582ms 440.927us           0 B           0 B           0 B           0 B            24  
     ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_tn         0.00%       0.000us         0.00%       0.000us       0.000us       9.831ms         5.17%       9.831ms 204.810us           0 B           0 B           0 B           0 B            48  
_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cu...         0.00%       0.000us         0.00%       0.000us       0.000us       8.631ms         4.54%       8.631ms 130.774us           0 B           0 B           0 B           0 B            66  
_ZN2at6native29vectorized_elementwise_kernelILi4EZNS...         0.00%       0.000us         0.00%       0.000us       0.000us       7.905ms         4.15%       7.905ms  14.971us           0 B           0 B           0 B           0 B           528  
_ZN7cutlass7Kernel2I68cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us       6.994ms         3.68%       6.994ms   2.331ms           0 B           0 B           0 B           0 B             3  
                                            aten::index         1.64%      24.623ms         2.38%      35.610ms     494.576us       6.386ms         3.36%       6.721ms  93.350us           0 B           0 B      52.12 MB      49.80 MB            72  
                                              aten::add         1.44%      21.594ms         2.35%      35.247ms     124.991us       6.113ms         3.21%       6.113ms  21.679us           0 B           0 B      90.38 MB      90.38 MB           282  
_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23...         0.00%       0.000us         0.00%       0.000us       0.000us       5.524ms         2.90%       5.524ms  34.744us           0 B           0 B           0 B           0 B           159  
                                aten::native_layer_norm         1.34%      20.062ms         1.85%      27.698ms     543.096us       4.752ms         2.50%       4.752ms  93.167us           0 B           0 B      64.15 MB           0 B            51  
_ZN2at6native53_GLOBAL__N__e58679f9_20_layer_norm_ke...         0.00%       0.000us         0.00%       0.000us       0.000us       4.752ms         2.50%       4.752ms  93.167us           0 B           0 B           0 B           0 B            51  
_ZN2at6native24vectorized_gather_kernelILi16ExEEvPcS...         0.00%       0.000us         0.00%       0.000us       0.000us       4.534ms         2.38%       4.534ms  52.117us           0 B           0 B           0 B           0 B            87  
                                              aten::mul         0.93%      13.902ms         1.64%      24.609ms      91.144us       4.476ms         2.35%       4.476ms  16.578us           0 B           0 B      90.85 MB      90.85 MB           270  
                                             aten::sort         0.72%      10.748ms         3.51%      52.636ms       4.386ms       4.384ms         2.30%       4.455ms 371.260us           0 B           0 B       2.25 MB       2.25 MB            12  
_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv...         0.00%       0.000us         0.00%       0.000us       0.000us       3.919ms         2.06%       3.919ms  81.640us           0 B           0 B           0 B           0 B            48  
_ZN2at6native29vectorized_elementwise_kernelILi4EZNS...         0.00%       0.000us         0.00%       0.000us       0.000us       3.847ms         2.02%       3.847ms  24.662us           0 B           0 B           0 B           0 B           156  
_ZN14at_cuda_detail3cub6detail10radix_sort29DeviceRa...         0.00%       0.000us         0.00%       0.000us       0.000us       3.650ms         1.92%       3.650ms  38.016us           0 B           0 B           0 B           0 B            96  
                                              aten::sub         0.27%       4.041ms         0.57%       8.507ms      45.737us       3.503ms         1.84%       3.503ms  18.836us           0 B           0 B     157.31 MB     157.31 MB           186  
_ZN2at6native27unrolled_elementwise_kernelINS0_15CUD...         0.00%       0.000us         0.00%       0.000us       0.000us       3.497ms         1.84%       3.497ms  48.571us           0 B           0 B           0 B           0 B            72  
                                     aten::_log_softmax         1.85%      27.645ms         2.44%      36.613ms      12.204ms       3.266ms         1.72%       3.266ms   1.089ms           0 B           0 B      48.00 MB      48.00 MB             3  
_ZN2at6native43_GLOBAL__N__bcfaaeff_10_SoftMax_cu_0a...         0.00%       0.000us         0.00%       0.000us       0.000us       3.266ms         1.72%       3.266ms   1.089ms           0 B           0 B           0 B           0 B             3  
_ZN2at6native20bitonicSortKVInPlaceILi2ELin1ELi16ELi...         0.00%       0.000us         0.00%       0.000us       0.000us       3.106ms         1.63%       3.106ms  47.058us           0 B           0 B           0 B           0 B            66  
                                     aten::masked_fill_         3.89%      58.350ms         5.03%      75.290ms     697.134us       3.055ms         1.61%       3.055ms  28.290us           0 B           0 B           0 B           0 B           108  
                                           aten::gather         0.43%       6.430ms         1.94%      29.044ms     225.148us       3.031ms         1.59%       3.031ms  23.493us           0 B           0 B      22.37 MB      22.37 MB           129  
                                            aten::fill_         0.83%      12.386ms         1.72%      25.812ms      55.154us       2.856ms         1.50%       2.856ms   6.103us           0 B           0 B           0 B           0 B           468  
_ZN2at6native24index_elementwise_kernelILi128ELi4EZN...         0.00%       0.000us         0.00%       0.000us       0.000us       2.771ms         1.46%       2.771ms 115.447us           0 B           0 B           0 B           0 B            24  
ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us       2.580ms         1.36%       2.580ms  53.743us           0 B           0 B           0 B           0 B            48  
                                            aten::where         0.91%      13.623ms         2.63%      39.346ms     252.219us       2.555ms         1.34%       4.443ms  28.481us           0 B           0 B      93.88 MB     -21.00 KB           156  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.498s
Self CUDA time total: 190.317ms

PROFILE REPORT: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase6_profile.json
CHROME TRACE: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase6_rlsa_trace.json
PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> py -m phases.phase6_profile --device cuda --amp bf16 --seq-len 1024 --mode flat_lsa
USDT:2026-11-09 00:22:27 24684:26260 C:\actions-runner\_work\pytorch\pytorch\third_party\kineto\libkineto\src\SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-11-09 00:22:30 24684:26260 C:\actions-runner\_work\pytorch\pytorch\third_party\kineto\libkineto\src\SyncActivityProfilerHandler.cpp:46] profiler_stop
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                               rlsa/reverse_index_build         0.00%       0.000us         0.00%       0.000us       0.000us     496.095ms       263.50%     496.095ms  82.682ms           0 B           0 B           0 B           0 B             6  
                    rlsa/batched_hierarchical_retrieval         0.00%       0.000us         0.00%       0.000us       0.000us     422.410ms       224.36%     422.410ms  35.201ms           0 B           0 B           0 B           0 B            12  
                        rlsa/summary_prefix_and_overlap         0.00%       0.000us         0.00%       0.000us       0.000us     141.539ms        75.18%     141.539ms  11.795ms           0 B           0 B           0 B           0 B            12  
                          rlsa/index_feature_projection         0.00%       0.000us         0.00%       0.000us       0.000us     104.543ms        55.53%     104.543ms  17.424ms           0 B           0 B           0 B           0 B             6  
                                               aten::mm         8.24%     189.956ms        13.36%     307.784ms       1.531ms      46.207ms        24.54%      46.207ms 229.886us           0 B           0 B     193.42 MB     193.42 MB           201  
                     aten::_efficient_attention_forward         0.16%       3.711ms         0.73%      16.928ms     564.253us      24.744ms        13.14%      25.037ms 834.574us           0 B           0 B      15.03 MB           0 B            30  
_Z40fmha_cutlassF_bf16_aligned_64x64_rf_sm80N22PyTor...         0.00%       0.000us         0.00%       0.000us       0.000us      24.744ms        13.14%      24.744ms 824.815us           0 B           0 B           0 B           0 B            30  
                                    rlsa/sdpa_attention         0.00%       0.000us         0.00%       0.000us       0.000us      22.955ms        12.19%      22.955ms   1.913ms           0 B           0 B           0 B           0 B            12  
                                            aten::copy_         2.89%      66.482ms         6.74%     155.210ms     139.452us      19.949ms        10.60%      19.949ms  17.924us           0 B           0 B           0 B           0 B          1113  
                                              aten::cat         1.25%      28.736ms         2.60%      59.835ms     443.221us      15.834ms         8.41%      15.834ms 117.285us           0 B           0 B     322.73 MB     322.73 MB           135  
_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22g...         0.00%       0.000us         0.00%       0.000us       0.000us      14.685ms         7.80%      14.685ms  66.150us           0 B           0 B           0 B           0 B           222  
                                             aten::topk         1.11%      25.541ms         3.89%      89.590ms       1.244ms      12.838ms         6.82%      12.838ms 178.301us           0 B           0 B       1.59 MB       1.59 MB            72  
ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us      11.918ms         6.33%      11.918ms 248.301us           0 B           0 B           0 B           0 B            48  
ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      10.582ms         5.62%      10.582ms 440.932us           0 B           0 B           0 B           0 B            24  
     ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_tn         0.00%       0.000us         0.00%       0.000us       0.000us       9.818ms         5.21%       9.818ms 204.548us           0 B           0 B           0 B           0 B            48  
_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cu...         0.00%       0.000us         0.00%       0.000us       0.000us       8.063ms         4.28%       8.063ms 149.324us           0 B           0 B           0 B           0 B            54  
_ZN2at6native29vectorized_elementwise_kernelILi4EZNS...         0.00%       0.000us         0.00%       0.000us       0.000us       7.908ms         4.20%       7.908ms  14.978us           0 B           0 B           0 B           0 B           528  
_ZN7cutlass7Kernel2I68cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us       6.997ms         3.72%       6.997ms   2.332ms           0 B           0 B           0 B           0 B             3  
                                            aten::index         1.11%      25.668ms         2.25%      51.863ms     720.317us       6.382ms         3.39%       6.715ms  93.257us           0 B           0 B      52.12 MB      49.80 MB            72  
_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23...         0.00%       0.000us         0.00%       0.000us       0.000us       5.885ms         3.13%       5.885ms  32.157us           0 B           0 B           0 B           0 B           183  
                                              aten::add         1.42%      32.666ms         2.29%      52.873ms     195.827us       5.850ms         3.11%       5.850ms  21.666us           0 B           0 B      90.35 MB      90.35 MB           270  
                                aten::native_layer_norm         0.87%      19.946ms         1.40%      32.242ms     632.192us       4.754ms         2.52%       4.754ms  93.207us           0 B           0 B      64.15 MB           0 B            51  
_ZN2at6native53_GLOBAL__N__e58679f9_20_layer_norm_ke...         0.00%       0.000us         0.00%       0.000us       0.000us       4.754ms         2.52%       4.754ms  93.207us           0 B           0 B           0 B           0 B            51  
_ZN2at6native24vectorized_gather_kernelILi16ExEEvPcS...         0.00%       0.000us         0.00%       0.000us       0.000us       4.538ms         2.41%       4.538ms  52.159us           0 B           0 B           0 B           0 B            87  
                                             aten::sort         0.47%      10.770ms         5.98%     137.789ms      11.482ms       4.339ms         2.30%       4.409ms 367.424us           0 B           0 B       2.25 MB       2.25 MB            12  
                                              aten::mul         0.87%      20.034ms         1.67%      38.492ms     156.472us       4.290ms         2.28%       4.290ms  17.439us           0 B           0 B      89.72 MB      89.72 MB           246  
_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv...         0.00%       0.000us         0.00%       0.000us       0.000us       3.922ms         2.08%       3.922ms  81.698us           0 B           0 B           0 B           0 B            48  
_ZN2at6native29vectorized_elementwise_kernelILi4EZNS...         0.00%       0.000us         0.00%       0.000us       0.000us       3.849ms         2.04%       3.849ms  24.671us           0 B           0 B           0 B           0 B           156  
_ZN14at_cuda_detail3cub6detail10radix_sort29DeviceRa...         0.00%       0.000us         0.00%       0.000us       0.000us       3.649ms         1.94%       3.649ms  38.013us           0 B           0 B           0 B           0 B            96  
                                              aten::sub         0.45%      10.366ms         0.97%      22.379ms     120.317us       3.507ms         1.86%       3.507ms  18.855us           0 B           0 B     157.31 MB     157.31 MB           186  
_ZN2at6native27unrolled_elementwise_kernelINS0_15CUD...         0.00%       0.000us         0.00%       0.000us       0.000us       3.501ms         1.86%       3.501ms  48.619us           0 B           0 B           0 B           0 B            72  
                                     aten::_log_softmax         1.32%      30.483ms         1.82%      41.927ms      13.976ms       3.266ms         1.73%       3.266ms   1.089ms           0 B           0 B      48.00 MB      48.00 MB             3  
_ZN2at6native43_GLOBAL__N__bcfaaeff_10_SoftMax_cu_0a...         0.00%       0.000us         0.00%       0.000us       0.000us       3.266ms         1.73%       3.266ms   1.089ms           0 B           0 B           0 B           0 B             3  
                                     aten::masked_fill_         2.26%      52.019ms         4.25%      97.858ms     815.480us       3.202ms         1.70%       3.202ms  26.680us           0 B           0 B           0 B           0 B           120  
                                            aten::fill_         0.78%      18.075ms         2.36%      54.460ms     116.367us       2.862ms         1.52%       2.862ms   6.115us           0 B           0 B           0 B           0 B           468  
_ZN2at6native20bitonicSortKVInPlaceILi2ELin1ELi16ELi...         0.00%       0.000us         0.00%       0.000us       0.000us       2.790ms         1.48%       2.790ms  51.668us           0 B           0 B           0 B           0 B            54  
_ZN2at6native24index_elementwise_kernelILi128ELi4EZN...         0.00%       0.000us         0.00%       0.000us       0.000us       2.771ms         1.47%       2.771ms 115.451us           0 B           0 B           0 B           0 B            24  
_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22g...         0.00%       0.000us         0.00%       0.000us       0.000us       2.668ms         1.42%       2.668ms  34.203us           0 B           0 B           0 B           0 B            78  
ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us       2.578ms         1.37%       2.578ms  53.701us           0 B           0 B           0 B           0 B            48  
                                            aten::where         0.86%      19.927ms         2.87%      66.175ms     424.201us       2.559ms         1.36%       4.449ms  28.516us           0 B           0 B      93.86 MB     -21.00 KB           156  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.304s
Self CUDA time total: 188.275ms

PROFILE REPORT: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase6_profile.json
CHROME TRACE: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase6_rlsa_trace.json
PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> 
