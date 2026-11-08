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

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py -m phases.phase1 --device cuda
PHASE 1 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase1_pass.json

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py -m phases.phase2 --device cuda
PHASE 2 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase2_pass.json

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py -m phases.phase3 --device cuda
PHASE 3 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase3_pass.json

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py -m phases.phase4 --device cuda
C:\Users\Kamal\AppData\Local\Programs\Python\Python311\python.exe: No module named phases.phase4

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py -m phases.phase4_train --device cuda --amp bf16 --stage1-steps 200 --stage2-steps 200 --seq-len-stage1 512 --seq-len-stage2 1024 --batch-size 1 --grad-accum 2
Small real-text dataset is missing. Run: python scripts/prepare_small_data.py

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py scripts/prepare_small_data.py
Loading Salesforce/wikitext/wikitext-103-raw-v1 ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Training 8000-token BPE on up to 50,000 text rows ...
[00:00:01] Pre-processing sequences       ██████████████████████████████████████████████████████████ 50000    /    50000[00:00:00] Tokenize words                 ██████████████████████████████████████████████████████████ 110326   /   110326
[00:00:00] Count pairs                    ██████████████████████████████████████████████████████████ 110326   /   110326
[00:00:01] Compute merges                 ██████████████████████████████████████████████████████████ 7740     /     7740
Encoding train, cap 5,000,000 tokens -> data\train.bin
  wrote 5,000,000 tokens (9.54 MiB)
Encoding validation, cap 200,000 tokens -> data\validation.bin
  wrote 200,000 tokens (0.38 MiB)
Encoding test, cap 200,000 tokens -> data\test.bin
  wrote 200,000 tokens (0.38 MiB)
DATA READY: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\data

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py -m phases.phase4_train --device cuda --amp bf16 --stage1-steps 200 --stage2-steps 200 --seq-len-stage1 512 --seq-len-stage2 1024 --batch-size 1 --grad-accum 2
Device: cuda | model parameters: 11,431,552 | vocab: 8000
Dataset tokens: train=5,000,000 val=200,000 test=200,000
stage1_dense step   50/200 | train 7.7419 | val 7.2919 | ppl 1468.4 | lr 2.61e-04 | tok/s 5804
stage1_dense step  100/200 | train 7.1182 | val 7.0263 | ppl 1125.9 | lr 1.62e-04 | tok/s 5725
stage1_dense step  150/200 | train 6.9249 | val 6.8688 | ppl 961.8 | lr 6.26e-05 | tok/s 5604
stage1_dense step  200/200 | train 6.9007 | val 6.8576 | ppl 951.1 | lr 2.00e-05 | tok/s 5568
stage2_rlsa step   50/200 | train 6.8205 | val 6.7926 | ppl 891.2 | lr 8.87e-05 | tok/s 2278 | remote 18.5 | overflow 0.376 | load 8.0 | activeA 27
stage2_rlsa step  100/200 | train 6.7448 | val 6.7697 | ppl 871.1 | lr 6.06e-05 | tok/s 2306 | remote 22.0 | overflow 0.236 | load 8.0 | activeA 29
stage2_rlsa step  150/200 | train 6.6933 | val 6.6811 | ppl 797.2 | lr 3.22e-05 | tok/s 2303 | remote 23.7 | overflow 0.165 | load 8.0 | activeA 28
stage2_rlsa step  200/200 | train 6.6870 | val 6.5371 | ppl 690.3 | lr 2.00e-05 | tok/s 2305 | remote 24.3 | overflow 0.135 | load 8.0 | activeA 30
PHASE 4 PASS: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase4_pass.json

C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized>py -m phases.phase5_benchmark --device cuda --amp bf16 --split test --seq-len 1024 --batch-size 1 --eval-batches 8
Checkpoint: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\runs\reverse_lsa_wt103_small\last.pt
Loaded stage=stage2_rlsa step=200 | params=11,431,552
Device=cuda | split=test | seq_len=1024 | batches=8
All modes use identical checkpoint weights and identical evaluation windows.
dense              | loss 6.68987 | ppl 804.22 | forward 34.96 ms | tok/s 29292 | VRAM 0.13602304458618164
swa                | loss 6.68942 | ppl 803.86 | forward 39.46 ms | tok/s 25949 | VRAM 0.13602304458618164
flat_lsa           | loss 6.68918 | ppl 803.67 | forward 92.34 ms | tok/s 11090 | VRAM 0.13604402542114258
rlsa               | loss 6.68937 | ppl 803.82 | forward 85.58 ms | tok/s 11966 | VRAM 0.13604402542114258
rlsa_no_summaries  | loss 6.68912 | ppl 803.61 | forward 82.71 ms | tok/s 12381 | VRAM 0.13604402542114258
PHASE 5 BENCHMARK COMPLETE: C:\Users\Kamal\Downloads\reverse_lsa_complete_optimized\artifacts\phase5_benchmark.json
This is a benchmark report, not yet a scientific PASS gate.
