# Transformer Encoder from Scratch — Sentiment Analysis Benchmark

Built a Transformer Encoder from scratch in PyTorch without using any pretrained models or high-level libraries, then benchmarked it against LSTM and GRU on the IMDB sentiment classification task.

## Results

| Model | Train Acc | Test Acc |
|-------|-----------|----------|
| Transformer | 93.2% | ~81% |
| **GRU** | **93.9%** | **83.9%** |
| LSTM | 53.6% | 52.0% |

**Winner: GRU** on this task and scale.

## Architecture

Built every component manually:

- `ScaledDotProductAttention` — Q, K, V dot product with √d_k scaling
- `MultiHeadAttention` — 4 parallel attention heads with learned projections
- `PositionalEncoding` — sinusoidal encoding (sin/cos at varying frequencies)
- `FeedForward` — two linear layers with ReLU
- `EncoderBlock` — attention + FFN with residual connections and LayerNorm
- `Encoder` — 2 stacked encoder blocks with token embedding
- `SentimentClassifier` — encoder + mean pooling + linear classification head

## Key Observations

**1. Transformer converges faster**
Transformer hit 80.9% test accuracy in epoch 1 while GRU was at 66.6%. This is the parallelization advantage — transformers process all tokens simultaneously via attention, LSTMs and GRUs process sequentially token by token.

**2. GRU wins at this scale**
With only 25k training samples and no pretraining, GRU matched and slightly beat the Transformer. Transformer's real advantage emerges at scale with massive data — this is why BERT and GPT are pretrained on billions of tokens before fine-tuning.

**3. Transformer overfits faster**
Transformer test accuracy peaked at epoch 2 (~81%) and degraded at epoch 3 as train accuracy hit 93%. Transformers have more parameters and need regularization techniques like dropout to generalize on small datasets.

**4. LSTM underperformed**
LSTM stayed near random (52%) across all epochs. Likely caused by vanishing gradients over long sequences (256 tokens) combined with a large vocabulary (251k words). Would likely recover with gradient clipping and a smaller vocab.

**5. Results have natural variance**
Without a fixed seed, test accuracy varies ~1-2% between runs due to random weight initialization. Seed is now fixed at 42 for full reproducibility.

## Dataset
- IMDB Movie Reviews — 25k train, 25k test
- Binary classification: positive / negative
- Vocab size: 251,639 unique tokens (space-split, lowercased)
- Max sequence length: 256 tokens

## Hyperparameters

| Param | Value |
|-------|-------|
| d_model | 128 |
| heads (h) | 4 |
| d_ff | 256 |
| N (encoder blocks) | 2 |
| max_seq_len | 256 |
| batch size | 32 |
| epochs | 3 |
| optimizer | Adam (lr=1e-3) |
| seed | 42 |

## What I'd do differently at scale
- Add dropout for better regularization
- Use BPE tokenizer to reduce vocab from 251k to ~30k
- Pretrain on a larger corpus before fine-tuning on IMDB
- Use learning rate warmup scheduler like the original paper
- Train for more epochs with early stopping on validation loss

## Note on Paper Config
The original "Attention is All You Need" paper used d_model=512, h=8, d_ff=2048, N=6 trained on millions of sentence pairs. This project intentionally scales down for fast training on a small dataset — the architecture is identical, only the scale differs.
