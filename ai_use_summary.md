# AI Use Summary

**Author:** Sijie Zhou
**Tool:** Claude (Anthropic), used through the claude.ai chat interface. The conversation was mostly in Chinese.

## Summary

Claude wrote the code and provided the overall outline of the project. It also explained each result I shared. I ran all experiments, and I wrote the content of the discussion and analysis sections myself, using Claude's outline and explanations as a guide.

## Overview

| Part of the project | What Claude did | What I did |
|---|---|---|
| Understanding the task | Explained the assignment and proposed the overall outline | Read the brief and followed the outline |
| Dataset | Suggested options and wrote the download and cleaning code | Chose the Sherlock Holmes books, ran the code and checked the statistics |
| Model and training code | Wrote all code: training loop, three models, generation, result saving | Ran every cell on Colab (T4 GPU) and checked the outputs |
| Comparison and attention code | Wrote the results table, loss plot, text metrics and attention tools | Ran them and reviewed the outputs |
| Interpreting results | Explained each output I shared | Decided which points to use |
| Per-model discussions | Provided the questions to answer and explained the samples | Wrote the content |
| Section 7.5 analysis | Provided the structure and a draft, which I did not use | Wrote the content |
| README | Provided the outline and a first draft | Reviewed and edited the text |

## How AI helped, step by step

**1. Outline.** I shared screenshots of the assignment. Claude explained that the task is a comparison of character-level language models. It proposed the outline I followed: prepare the data once, share one training loop, add the three models one by one, then compare them.

**2. Dataset.** Claude first suggested Tiny Shakespeare. I asked for a less common dataset and chose the Holmes books from the options it gave. Claude wrote the code to download them, remove the Gutenberg licence text, encode characters and split the data.

**3. Code.** Claude wrote all of the code in the notebook. I did not change it. Claude's own environment had no PyTorch, so it never ran the code. Every result in this repository comes from my runs on Google Colab.

**4. Interpreting results.** After each step, I sent Claude screenshots of the outputs. Claude explained what the samples, losses, curves, quote counts and attention weights meant.

**5. Writing.** For each written section, Claude provided the structure and the questions to answer. I wrote the content of the three discussion sections and the answers in section 7.5. Claude drafted the README from my notebook, and I reviewed it.

## What I did myself

- Chose the dataset.
- Ran the full notebook on Colab and saved all results.
- Checked the sanity tests in the outputs. Each model's starting loss was close to ln(91) ≈ 4.51, and the parameter counts matched what was expected.
- Shared each output with Claude and asked questions about it.
- Decided not to run an extra 10,000-step experiment.
- Wrote the discussion sections and the section 7.5 answers.

## Concepts I worked through

- How the bigram model works as a 91 × 91 lookup table, and why it cannot spell long words.
- Why the training and validation split is sequential, not random.
- How the LSTM carries earlier context in a hidden state, and why that limits long-range memory.
- How causal self-attention works: queries, keys, values and the lower-triangular mask.
- How to read loss curves to tell overfitting from undertraining.
- What the attention maps showed about the roles of different layers.

## Limits of AI help

Claude could not run the code, so it saw no results until I shared them. Some of its early estimates were wrong. For example, it expected the LSTM's validation loss to be between 1.5 and 1.8, but the real value was 1.15. I treated its explanations as guidance and used my actual outputs as the final source.
