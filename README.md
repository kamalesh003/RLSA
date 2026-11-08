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
PS C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized> 
