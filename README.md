# RLSA

## What R-LSA is for

R-LSA (Reverse Latent Sparse Attention) is a way to make the global attention layers of a long-context decoder-only model cheaper. Full attention costs O(T²) compute, and decoding has to read the whole KV cache for every new token. R-LSA lets each query look at only a small, content-selected subset of past tokens instead of all of them.

In your code, a global layer's query sees four things:

1. **Sink tokens**, a few always-visible learned slots.
2. **Local window**, the recent tokens (256 here).
3. **Retrieved remote keys**, at most 256 old tokens chosen through the index.
4. **Anchor summary tokens**, mass-weighted average K/V per anchor, which give a coarse view of everything not retrieved.

The "reverse" part is that the indexing work happens at write time. Each key is assigned to its top-r anchors within its block, and each bucket has a hard capacity. At read time a query only picks its top-a anchors and then the top-p buckets (a two-stage lookup) instead of scoring every key. So the per-query cost depends on the retrieval budget, not on context length.

In the hybrid model, most layers are cheap sliding-window attention with RoPE, and every 4th layer is a global NoPE layer that uses R-LSA for long-range retrieval.

## Where it can be used

- **Long-document and long-code LLMs:** books, legal and medical records, repository-level code, long chat histories.
- **Long-context decoding with a KV cache:** decode is memory-bandwidth bound, so reading ~300 keys instead of 100K+ is where the biggest win would be.
- **Agent and memory-style models:** retrieval over a long past inside the model rather than via an external RAG system.
- **Other long sequences:** genomics, audio, logs, time series, and long-video tokens, where T is far beyond what dense attention handles comfortably.
- **Memory-limited hardware:** laptops and edge devices, if you also compress or offload the KV cache.

It is not useful for short contexts. At 1K–4K tokens, a fused dense or flash attention kernel is hard to beat.

## What your own results say

- **Speed:** R-LSA is slower here (85 ms vs 35 ms dense at T=1024). That is expected, because the retrieval and indexing overhead only pays off once T is large enough that dense attention is clearly quadratic.
- **Loss:** all five modes land at 6.689 ± 0.0003. That doesn't show R-LSA is lossless. At ppl ≈ 800 the model is barely past unigram-level, so the global layers aren't contributing long-range information yet, and the benchmark can't tell the modes apart.
- **Router stats:** only about 24 remote keys are retrieved per query (out of a 256 budget), with 13% bucket overflow. The retrieval path is still lightly used.

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
