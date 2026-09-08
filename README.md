# attention-all-you-need
# minicodeGPT

A from-scratch **Encoder–Decoder Transformer** for **Hindi → English machine translation**, implemented in PyTorch without using a pretrained Transformer model.

The goal of this project is to understand and implement the core architecture behind modern sequence-to-sequence Transformers, including multi-head attention, self-attention, cross-attention, positional encoding, masking, teacher forcing, and autoregressive inference.

---

## 🚀 Project Overview

This project implements a complete Transformer-based translation pipeline:

```text
Hindi Sentence
      ↓
BPE Tokenization
      ↓
Token Embedding
      ↓
Positional Encoding
      ↓
Encoder
      ↓
Contextual Representations
      ↓
Decoder
      ↓
Autoregressive Translation
      ↓
English Sentence



## 📊 Results

The model was trained on approximately 49.8K Hindi-English
translation pairs.

### Training

| Metric | Result |
|---|---:|
| Training pairs | ~44.8K |
| Validation pairs | ~5K |
| Epochs | 10 |
| Model dimension | 128 |
| Transformer layers | 2 |
| Attention heads | 8 |

### Sample translations

| Hindi | Model Output |
|---|---|
| मैं एक सेब खाता हूँ | ... |
| मुझे किताबें पसंद हैं | ... |
| मैं स्कूल जाता हूँ | ... |
