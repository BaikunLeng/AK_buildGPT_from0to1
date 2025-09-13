# AK_buildGPT_from0to1

A minimal implementation of a GPT-style language model, built step by step.  
This project follows Andrej Karpathy's "GPT from scratch" style, with extra benchmarking on multiple datasets and integration with Weights & Biases (W&B).

---

## Project Structure
AK_buildGPT_from0to1/
gpt_mini.py         # Main training script (tiny GPT-like model)
run_test.sh         # Helper shell script to run experiments on different datasets
input.txt           # Tiny Shakespeare dataset (downloaded automatically if missing)
data_ptb/           # Penn Treebank dataset (train/valid text files)
wandb/              # Auto-generated logs from W&B runs

---

## Datasets

This repo supports 3 datasets:

1. **Tiny Shakespeare**  
   - Source: [karpathy/char-rnn](https://github.com/karpathy/char-rnn)  
   - Small character-level dataset (~1 MB), good for fast experiments.  
   - Loaded from `input.txt`.

2. **WikiText-2 (raw)**  
   - Source: HuggingFace datasets (`wikitext`, `wikitext-2-raw-v1`)  
   - A medium-sized English dataset with real sentences.

3. **Penn Treebank (PTB)**  
   - Fallback via direct download (from Tom Sercu’s repo).  
   - Stored under `data_ptb/`.

---

## Training Script

The main entry point is:

```bash
python3 gpt_mini.py --dataset <tinyshakespeare|wikitext2|ptb>
