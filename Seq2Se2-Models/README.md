# Grapheme-to-Phoneme Conversion Using Seq2Seq Models

A character-level sequence-to-sequence (Seq2Seq) neural network for English grapheme-to-phoneme (G2P) conversion, implemented in PyTorch.

```
through  →  TH R UW
```

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameter Search](#hyperparameter-search)
- [Results](#results)
- [Usage](#usage)
- [Potential Extensions](#potential-extensions)

---

## Overview

This project implements and compares three decoder architectures for G2P conversion:

| Architecture | Description |
|---|---|
| **No Attention** | Decoder uses only the encoder's final hidden state |
| **Fixed Context** | Same encoder context vector fed at every decoding step |
| **Cross-Attention** | Decoder attends over all encoder hidden states |

**Key objectives:**
- Implement a custom multi-layer LSTM from scratch using manually defined LSTM cells
- Build a full encoder-decoder architecture for sequence transduction
- Evaluate and compare context mechanisms using multiple metrics
- Analyse performance as a function of input word length

---

## Project Structure

```
project/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── notebooks/
│   └── seq2seq_g2p.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install torch pandas matplotlib nltk numpy
```

**Required packages:** PyTorch · Pandas · NumPy · Matplotlib · NLTK

---

## Dataset

CSV files with two columns:

| word | phonemes |
|---|---|
| cat | K AE T |
| phone | F OW N |
| through | TH R UW |

### Preprocessing

Separate vocabularies are built for characters and phoneme tokens, with the following special tokens:

| Token | Purpose |
|---|---|
| `<PAD>` | Padding |
| `<SOS>` | Start of sequence |
| `<EOS>` | End of sequence |
| `<UNK>` | Unknown token |

**Example encoding:**

```
Word:     "cat"  →  [<SOS>, c, a, t, <EOS>, <PAD>, ...]
Phonemes: "K AE T"  →  [<SOS>, K, AE, T, <EOS>, <PAD>, ...]
```

---

## Model Architectures

### Encoder

- Embeds input characters
- Processes them through a stacked LSTM
- Outputs all hidden states and the final hidden/cell states

**Recurrence:**

$$h_t, c_t = \text{LSTMCell}(x_t, h_{t-1}, c_{t-1})$$

### Decoder

- Initialised from encoder final states
- Receives `<SOS>` as the first input token
- Predicts one phoneme at a time until `<EOS>`

### Cross-Attention

$$e_{t,s} = h_t^{\text{dec}} \cdot h_s^{\text{enc}}$$

$$\alpha_{t,s} = \text{softmax}(e_{t,s})$$

$$c_t = \sum_s \alpha_{t,s}\, h_s^{\text{enc}}$$

---

## Training

### Teacher Forcing

```
Full Target:    <SOS> K AE T <EOS>
Decoder Input:  <SOS> K AE T
Expected Label: K AE T <EOS>
```

### Loss Function

$$\mathcal{L} = -\sum_t \log P(y_t^* \mid y_{<t}, x)$$

### Optimisation Settings

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing |
| Label Smoothing | 0.1 |
| Gradient Clipping | 1.0 |

---

## Evaluation

| Metric | Description |
|---|---|
| **Token Accuracy** | % of correctly predicted non-padding tokens |
| **Word Accuracy** | Entire predicted sequence must match target exactly |
| **Phoneme Error Rate (PER)** | Normalised Levenshtein edit distance |
| **Validation Loss** | Cross-entropy under teacher forcing |

### Length-Based Analysis

Test words are bucketed by character count to evaluate performance vs. sequence length:

- 1–4 characters
- 5–7 characters
- 8–10 characters
- 11+ characters

---

## Hyperparameter Search

Random search over 27 configurations (3 × 3 × 3):

```python
LEARNING_RATES = [1e-5, 1e-4, 1e-3]
EMBED_SIZE     = [32, 128, 256]
HIDDEN_SIZE    = [64, 128, 256]
```

Best configuration selected by validation word accuracy.

---

## Results

| Model | Token Accuracy | Word Accuracy | PER |
|---|---|---|---|
| No Attention | 0.994 | 0.929 | 0.012 |
| Fixed Context | ~0.80–0.95 | Lower | Higher |
| Cross-Attention | ~0.98–0.99 | High | Low |

**Why No-Attention performs well:** G2P is largely monotonic — each grapheme maps to nearby phonemes, and words are short enough that the final encoder state retains sufficient information.

**Why Attention helps:** Attention improves robustness for long words, irregular spellings, and complex pronunciation patterns.

**Why Fixed Context underperforms:** Feeding a constant vector at every decoding step introduces redundancy and optimisation difficulties.

---

## Usage

```python
word = "through"
prediction = trainer.predict(word)
print(prediction)
# TH R UW
```

---

## Key Components

| Category | Components |
|---|---|
| Data Utilities | `read_csv_data`, `build_vocab`, `encode_word`, `decode_tokens` |
| Dataset | `Seq2SeqDataset` |
| Models | `LSTMCell`, `LSTMModel`, `Encoder`, `Decoder`, `Seq2Seq` |
| Training | `ModelTrainer` |
| Analysis | `plot_compare_results`, `random_hyperparam_search`, `bucket_words_by_length` |

---

## Potential Extensions

- Bidirectional encoder
- Beam search decoding
- Scheduled sampling
- Bahdanau attention
- Transformer-based models
