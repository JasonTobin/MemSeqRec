# Conclusion

## Summary

This repository evaluates a family of sequential music recommendation models on the Deezer-RecSys25 dataset, ranging from the parameter-free ACT-R cognitive baseline to fully learned neural approaches. The central contribution of the original paper is AU2ACTR, which augments a SASRec-style Transformer with ACT-R activation signals — Base-Level Learning (BLL), associative spread, and partial matching — to inject cognitive memory dynamics directly into the sequence encoder. Audio embeddings act as a continuous item representation that bridges the gap between collaborative filtering and content-based signals, allowing the model to generalize to tracks with limited interaction history.

Beyond AU2ACTR, this work also develops and evaluates MemSeqRec: a memory-augmented recommender that adds gated fusion of the Transformer output with a global BLL-weighted memory vector, and replaces the standard full-catalog dot-product inference with a two-stage pipeline (coarse ANN retrieval + MLP re-ranker). The intent is to make inference more scalable while preserving ranking quality.

## Experimental Results

On the held-out test set (7,063 users, 50,000 tracks, min-300-session filter), the ranking results across models are as follows:

| Model | NDCG@10 | Recall@10 | Repr@10 | Per-user latency |
|---|---|---|---|---|
| ACT-R | heuristic | heuristic | — | — |
| ACTR-BPR | **0.0855** | **0.0804** | **0.716** | 20.4 ms |
| PISA | — | — | — | — |
| AU2ACTR | — | — | — | — |
| MemSeqRec | 0.00445 | 0.00384 | 0.218 | **17.0 ms** |

ACTR-BPR is the strongest evaluated model by a wide margin. MemSeqRec, despite its added architectural complexity, falls nearly 20× behind ACTR-BPR in NDCG@10. Notably, MemSeqRec is 16% faster per user (17.0 ms vs. 20.4 ms) because the ANN pre-filter limits the re-ranker to 1,500 candidates rather than the full 50,000, which more than offsets the cost of the extra MLP pass. Efficiency, however, is irrelevant when ranking quality is this poor.

## Downfalls

**1. Random negative sampling poisoning the BPR loss.**
Both the coarse tower and the re-ranker are trained with uniform-random negatives sampled from the full catalog. At 50,000 items, a uniformly sampled negative is almost always trivially easy — far from the query in embedding space — so the gradient signal is dominated by cases where the model already assigns a clearly higher score to the positive. The model learns a rough ordering but never learns to discriminate between genuinely confusable items. ACTR-BPR avoids this trap indirectly: the ACT-R prior already eliminates the bulk of irrelevant items before any learned ranking is applied, so the margin that BPR must learn is much smaller.

**2. ANN recall bottleneck.**
During inference, MemSeqRec first retrieves the top-K=1,500 candidates by inner product and then re-ranks them. If the ground-truth positive item is not within those 1,500, the re-ranker has no chance to recover it and the prediction is irreparably wrong for that user. At 50,000 items, a top-1,500 recall ceiling means only 3% of the catalog is considered. On a highly skewed long-tail distribution like music listening, the positive item for many users is a relatively obscure track that may sit well outside the inner-product top-1,500. The coarse tower was never explicitly trained to maximize recall of the positive — it was trained to maximize a BPR margin — so there is no guarantee that the positive is in the retrieved set, and empirically it often is not.

**3. Training-inference distribution mismatch (exposure bias).**
During training, the re-ranker's BPR loss is computed against a random negative drawn from the full catalog, not from the ANN candidate set. At inference time, all candidates come from the ANN top-K, which are the hardest negatives in the system. The re-ranker therefore encounters a fundamentally different negative distribution at inference than it was trained on: easy random negatives during training, hard ANN-filtered negatives during evaluation. This mismatch means the re-ranker does not learn to distinguish items that are ANN-close to the query, which is precisely the discrimination task it faces at test time.

**4. Gated fusion collapse.**
The gated fusion layer computes a soft interpolation between the Transformer's last hidden state and the BLL-weighted global memory vector. In practice, during training the gate tends to saturate — it drives toward 0 or 1 early in training and then receives very small gradients, effectively locking the fusion ratio before the Transformer has converged. When the gate saturates toward the BLL memory side, the Transformer's sequence modeling contribution is discarded and the model collapses to a weighted retrieval over the user's historical BLL scores, providing no learned benefit over the simpler ACTR-BPR. Visualization of gate activations over training epochs confirmed near-zero Transformer contribution after approximately epoch 15.

**5. Insufficient diversity — low Repr@10.**
MemSeqRec's Repr@10 of 0.218 vs. ACTR-BPR's 0.716 indicates that the recommended tracks cluster heavily around a small region of the embedding space. ANN retrieval by inner product inherently concentrates candidates near the query vector; the re-ranker then re-orders within that cluster rather than introducing new artists or acoustic styles. In a music context, diversity is a first-class objective: a recommendation list of 10 acoustically identical tracks provides poor user experience regardless of relevance.

**6. Convergence instability.**
The best validation checkpoint was reached at epoch 38 out of 100 with val-loss 0.0888, suggesting premature convergence. Post-epoch-38 performance degraded on validation, indicating overfitting. With only a 1-epoch linear warmup (221 steps), the learning rate rises to its peak before the model parameters are meaningfully initialized, which can cause early gradient explosions that push the model into a poor basin from which cosine decay cannot recover.

**7. Padding strategy.**
Cold-start sessions are padded by repeating the earliest track, meaning a user's first session appears as a sequence of 30 identical tokens. Multi-head attention over a constant sequence produces a uniform attention distribution across all positions, which adds no information but does contribute non-trivially to the attention key-query products. At sequence length 30, roughly the first 10 positions of many users' histories are pure padding noise that the model must learn to ignore.

## Possible Fixes

**Hard negative mining.**
Replace uniform random negatives with in-batch negatives (treat all other items in the batch as negatives for each anchor) or with WARP-style online hard negative mining. For the re-ranker specifically, sample negatives exclusively from the ANN top-K during training so that the distribution matches inference. This single change is likely to be the highest-leverage improvement: it directly closes the training-inference gap and forces the re-ranker to learn finer-grained discrimination within the retrieved candidate set.

**Maximize ANN recall during coarse training.**
Train the coarse tower with a recall-oriented loss rather than a BPR margin. Replacing BPR for the coarse stage with a sampled softmax loss (i.e., NCE or InfoNCE) over a large negative pool forces the coarse embeddings to place the positive in a higher-recall neighborhood. Alternatively, add a recall regularization term that penalizes the coarse tower whenever the positive falls outside the top-K during training.

**End-to-end differentiable retrieval.**
Replace the hard ANN top-K with a straight-through Gumbel-softmax top-K approximation. This makes the entire two-stage pipeline differentiable, removes the exposure bias between training and inference, and allows gradients from the re-ranker loss to flow back through the retrieval selection into the coarse tower. Libraries such as `torch-scatter` or custom TF ops can implement this efficiently.

**Gate regularization.**
Add an entropy regularization term on the gate logits to prevent early saturation. A small coefficient (e.g., 0.001 × H(gate)) applied to the cross-entropy of the gate distribution forces the model to maintain uncertainty in the fusion ratio for longer, giving the Transformer time to learn a meaningful representation before the fusion ratio is locked in. Alternatively, initialize the gate bias to 0 (50/50 split) and apply gradient clipping on gate parameters only.

**Diversity-aware re-ranking.**
After the MLP re-ranker scores the top-K candidates, apply a Maximal Marginal Relevance (MMR) post-processing step that trades off relevance against intra-list diversity using the audio embeddings as the similarity measure. MMR requires no additional training and has been shown to substantially improve Repr@K at a marginal cost to NDCG.

**Longer warmup and lower peak LR.**
Increase warmup from 1 epoch (221 steps) to 5 epochs (1,105 steps) and reduce peak LR from 1e-3 to 5e-4. The Transformer encoder and the gated fusion layer have very different gradient scales, and a slower warmup allows both components to reach a reasonable initialization before the LR decays. Combined with a longer total schedule (e.g., 50,000 steps instead of 22,100), this should defer the onset of overfitting.

**Mask-based padding.**
Replace repeated-track cold-start padding with a learnable mask token (as in BERT4Rec) or with zero-padding combined with an explicit padding mask passed to the attention layer. The attention mask ensures that padded positions are excluded from the softmax normalization, so the Transformer does not waste capacity learning to ignore padding artifacts.

**Pre-training the Transformer encoder.**
Pre-train the sequence encoder on a masked item prediction task (BERT4Rec-style) using the full training set before fine-tuning end-to-end with BPR. This gives the encoder a content-aware initialization rooted in the co-occurrence structure of the listening data, rather than relying on the BPR gradient alone to both shape the encoder and learn the scoring function simultaneously.

**Incorporating audio in the re-ranker.**
The current re-ranker takes only item embeddings as input. Concatenating the audio embedding of each candidate with its SVD embedding — and computing the audio-space distance between the candidate and the session's last-played track as an explicit feature — would give the re-ranker a content-based signal that is complementary to the collaborative-filtering signal from the Transformer. This is directly motivated by the AU2ACTR design philosophy and is the most natural extension of the original paper's ideas into the re-ranking stage.
