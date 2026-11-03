# Character-Level Language Models: Bigram vs. LSTM vs. Transformer

**Author:** Sijie Zhou

This project trains three language models on the same text and compares them. All three solve one task: predict the next character from the previous context.

1. **Bigram model** — uses only the current character.
2. **LSTM** — reads the context from left to right and keeps a hidden state.
3. **Small transformer** — uses causal self-attention to look at earlier characters directly.

All code, results and analysis are in `char_lm_comparison.ipynb`.

## Repository structure

```
.
├── README.md                    # This file
├── char_lm_comparison.ipynb     # Data, models, training, comparison, analysis
├── ai_use_summary.md            # How AI was used in this project
├── data/
│   ├── holmes.txt               # Cleaned dataset (created by the notebook)
│   └── raw/                     # Original Gutenberg files
├── results/
│   ├── bigram.json              # Final losses, time, settings, loss history
│   ├── lstm.json
│   ├── transformer.json
│   ├── results_table.md         # Comparison table
│   ├── loss_curves.png          # Training and validation loss
│   └── attention_layer3.png     # Attention heatmaps (layer 3)
└── samples/
    ├── bigram.txt               # 500-character samples
    ├── lstm.txt
    ├── transformer.txt
    └── *_long.txt               # 3,000-character samples
```

## How to run

The notebook was run on Google Colab with a Tesla T4 GPU.

1. Open `char_lm_comparison.ipynb` in Colab.
2. Choose *Runtime → Change runtime type → T4 GPU*.
3. Choose *Runtime → Run all*.

The notebook downloads the data on its first run. Training all three models takes about 5 minutes on a T4. It needs `torch`, `pandas` and `matplotlib`, which Colab already includes. It also runs on a CPU, but more slowly.

## Dataset

The dataset is made of two short-story collections by Arthur Conan Doyle from Project Gutenberg. Both are in the public domain in the US.

| Book | Gutenberg ID | Link |
|---|---|---|
| *The Adventures of Sherlock Holmes* | 1661 | https://www.gutenberg.org/ebooks/1661 |
| *The Memoirs of Sherlock Holmes* | 834 | https://www.gutenberg.org/ebooks/834 |

**Download.** The notebook downloads both files automatically. If that fails, download the "Plain Text UTF-8" version of each book from the links above. Save them as `data/raw/adventures.txt` and `data/raw/memoirs.txt`, then run the notebook again.

**Cleaning.** The notebook removes the Gutenberg licence text before `*** START OF ... ***` and after `*** END OF ... ***`. It then joins the two books into `data/holmes.txt`.

**Statistics.**

| Item | Value |
|---|---|
| Total characters | 1,131,757 |
| Vocabulary size | 91 characters |
| Training set (first 90%) | 1,018,581 characters |
| Validation set (last 10%) | 113,176 characters |

The split is sequential, not random. Training uses random chunks of 128 characters, so a random split would let training and validation chunks overlap.

The text uses curly quotes (“ ” ‘ ’) and no straight quotes. I kept them because “ and ” are different characters. To close a quote correctly, a model has to remember whether it is inside a dialogue. This makes quotes a simple test of long-range memory.

## Models and settings

| | Bigram | LSTM | Transformer |
|---|---|---|---|
| Structure | 91 × 91 lookup table | Embedding (128) → 2-layer LSTM (256) → linear | Token + position embedding (128) → 4 blocks → LayerNorm → linear |
| Context used | Current character only | Up to 128 characters, through a hidden state | Up to 128 characters, through self-attention |
| Parameters | 8,281 | 956,635 | 831,579 |
| Learning rate | 1e-2 | 2e-3 | 1e-3 |
| Dropout | — | 0.2 | 0.2 |

Each transformer block has 4-head causal self-attention and a feedforward layer (128 → 512 → 128, GELU). Both parts use pre-LayerNorm and residual connections. The attention is written by hand so each step can be inspected.

**Shared settings.** All three models used the same data split and the same training loop. Each one trained for 5,000 steps with AdamW, a batch size of 64 and a context length of 128. Losses were averaged over 200 random batches every 500 steps. The random seed was 42. Training time counts only training steps, not evaluation.

The learning rate was set for each model on its own, since a value that works for a lookup table is too large for a deep network. The LSTM and transformer have a similar number of parameters, so the comparison is mainly about structure, not size.

## Results

### Loss and time

| Model | Params | Train loss | Val loss | Val perplexity | Gap (val − train) | Train time (s) |
|---|---|---|---|---|---|---|
| Bigram | 8,281 | 2.4229 | 2.3783 | 10.79 | −0.0446 | 16.7 |
| LSTM | 956,635 | 1.0715 | **1.1542** | **3.17** | 0.0827 | 118.1 |
| Transformer | 831,579 | 1.1978 | 1.1935 | 3.30 | −0.0043 | 152.9 |

All runs used a Tesla T4 GPU. Perplexity is `exp(val loss)`. Roughly, it is the number of characters the model is choosing between at each step. Before training, every model starts near ln(91) ≈ 4.51, which is a uniform guess.

![Loss curves](results/loss_curves.png)

### Text quality

Each model generated 3,000 characters from the same seed.

| Model | Real-word rate | “ count | ” count |
|---|---|---|---|
| Bigram | 37.5% | 8 | 12 |
| LSTM | 91.7% | 2 | 3 |
| Transformer | 89.4% | 11 | 10 |

*Real-word rate* is the share of generated words that also appear in the training text. For reference, the training text has about 13.6 opening quotes per 3,000 characters. Opening quotes also outnumber closing quotes in the books (5,126 vs. 4,027). This is because a speech that runs over several paragraphs opens each paragraph with “ but closes only once. A small gap between the two counts is therefore normal.

### Sample excerpts

Full samples are in `samples/`.

**Bigram**

```
d cre sse cr amyougeta sustond,
quthescave-nty nthefrmaingrer?”
was bvas I   or, Ba alon  I fascat urete ofise by.
```

**LSTM**

```
“They were that, horrorse crops I get away to the other.”

“Prentry, married Streathred,” said I, ‘yes, as it look upon the uppose,
is introduced in the straggled towards the crime, and my wife has been my sherry
```

**Transformer**

```
“I’re I have a sure adverything very remarkable thing?”

“Why, told you well be serious,” said Holmes.
```

## Analysis

**Which model performed best?**
The LSTM had the lowest validation loss (1.1542). The transformer was close (1.1935), and the bigram model was far behind (2.3783). However, the curves show different trends. After about step 2,000, the LSTM's training loss kept falling while its validation loss levelled off, so it was starting to overfit. The transformer's two curves stayed close and were still going down at step 5,000. Under this 5,000-step budget, the LSTM was best, but the transformer had more room to improve.

The bigram's validation loss was lower than its training loss. A bigram cannot overfit, so this suggests the validation text is slightly easier than the training text.

**Which model trained fastest?**
The bigram model (16.7 s), since each step is only a table lookup. The LSTM took about 7 times as long (118.1 s). The transformer was the slowest (152.9 s, about 30% slower than the LSTM). PyTorch's LSTM uses an optimised cuDNN kernel, while my attention code is a set of separate matrix operations. At this small scale, the parallel nature of attention did not make up for that.

**Which model generated the most realistic text?**
At the word level, the LSTM was slightly better, with 91.7% real words against 89.4% for the transformer. The bigram model reached only 37.5%. The transformer, however, was better at dialogue. It wrote 11 opening and 10 closing quotes, which is close to the dialogue density of the books. The LSTM wrote only 2 and 3. In one LSTM sample, a quote opened with ‘ but closed with ”. None of the three models produced sentences with real meaning. These results come from one random seed.

**What did the transformer improve compared to the other models?**
It did not lower the loss in this experiment. It can look at any earlier character in the 128-character window in one step, while the LSTM must pass that information through its hidden state. It also generalised better, with almost no gap between training and validation loss. Finally, it kept dialogue structure better: paragraphs open and close with quotes, and lines like “Why, told you well be serious,” said Holmes. follow the books' format.

**Was the improvement worth the added complexity?**
Not at this scale. The transformer needed more code, trained more slowly and ended with a slightly higher validation loss. If the goal is the lowest loss after a few minutes on one million characters, the LSTM is the better choice. Even so, the trend favours the transformer. It showed no overfitting and was still improving, while the LSTM had almost stopped. With more data or longer training, the transformer would probably pay off. I did not test this, so it is an expectation based on the curves.

**What did you learn about attention?**
Attention lets each position choose, based on content, which earlier positions to read from. The attention maps show that different layers do different jobs.

- In layer 0, head 2 put 100% of its weight on the previous character in both test sentences. Head 1 did the same for the character two steps back. These heads work like a built-in n-gram model.
- Higher layers look by content. In `“I have seen it before,” said Holmes. “It is quite clear.`, the model must predict what follows the final full stop. Layer 2 head 1 gave 99% of its attention to the opening quote 18 characters earlier, and layer 3 head 0 gave it 92%. Layer 3 head 2 focused on the previous closing quote (98%). The model seems to check whether it is inside a dialogue.
- With `Holmes looked at me. I said nothing, and Hol`, no head copied from the first "Holmes". The model mostly looked at the current word. It has probably memorised that "Hol" is followed by "mes", so it does not need to look back.

Attention also has limits. It cannot see beyond the 128-character window, and attention weights show where a model looks, not why it makes a prediction.

## Reflection

The biggest step was from the bigram model to the two neural models. Perplexity fell from about 10.8 to about 3.2, which shows that using more context matters far more than the exact way it is used.

I expected the transformer to win, but the LSTM had the lower loss. Looking at the loss curves helped me understand why. The LSTM learned faster at the start, since reading text in order suits it well. By the end, though, it was overfitting, while the transformer was still improving. One final number does not tell the whole story.

Generated text was hard to judge by eye. Two short samples could look equally good or bad. Simple measures, such as the real-word rate and quote counts, gave a clearer picture. Even these had limits. For example, the books themselves do not have equal numbers of opening and closing quotes.

The attention maps were the most useful part for me. They turned "attention looks at earlier positions" into something I could see: some heads look at fixed positions, and others jump back to a quote mark. I also learned that a test can fail to show what I expected, as with the "Holmes" example, and that this is still a valid result.

## AI use

See [`ai_use_summary.md`](ai_use_summary.md).
