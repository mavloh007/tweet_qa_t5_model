# TweetQA — Full Fine-Tuning vs LoRA on T5

## Overview

This repo trains and evaluates a sequence-to-sequence model on the **TweetQA** dataset, comparing **full fine-tuning** vs **LoRA (parameter-efficient fine-tuning)**.

* **Dataset:** TweetQA (social media QA).
  Hugging Face: [https://huggingface.co/datasets/ucsbnlp/tweet_qa](https://huggingface.co/datasets/ucsbnlp/tweet_qa)
* **Models:** `google/flan-t5-small` 
* **Metrics:** token F1, semantic similarity.

## What’s inside

* `tweet_qa_main.py` — full fine-tuning pipeline.
* `twet_qa_main_lora.py` - LoRA version (same training loop; only model wiring differs).
* `helper_functions.py` — dataset class, collate fn, metrics, evaluation, `answer()` helper.
* `requirements.txt` — dependencies.

---

## 1) Install requirements

```bash
python -m venv .venv && source .venv/bin/activate   # (optional but recommended)
pip install -r requirements.txt
```

**requirements.txt** (key libs):

* `torch`, `transformers`, `datasets`, `sentence-transformers`, `tqdm`, `pandas`, `matplotlib`, `peft` (for LoRA)

---

## 2) Train

### A) Full fine-tune

```bash
python main.py \
  --model_name google/flan-t5-small \
  --epochs 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_dir flan_t5_tweetqa_ckpt
```

### B) LoRA fine-tune

```bash
python tweetqa_main_lora.py \
  --model_name google/flan-t5-small \
  --epochs 5 \
  --batch_size 8 \
  --lr 3e-4 \
  --save_dir flan_t5_tweetqa_ckpt_lora
```

> Notes
>
> * LoRA script wraps the base model with PEFT (`LoraConfig(r=8, alpha=32, dropout=0.05, target_modules=["q","v"])`) and optimizes only adapter params.
> * Everything else (loader, evaluation, plotting) is identical.

---

## 3) What gets saved

Each run writes to `--save_dir`:

* **Model checkpoint**
  * Full FT: base model weights + tokenizer files.
  * LoRA: **adapters only** + tokenizer files (load with the same base model).
    
* **Metrics & diagnostics**
  * Training/validation loss curves and validation F1/semantic-sim plots 
  * Console logs include: epoch time, steps/sec, peak VRAM.

---

## 4) Evaluate during training

The script evaluates on the **validation split** each epoch and prints:

* `val_loss`
* `F1`
* `avg_sem_sim` 

You can switch to faster epoch checks with `beams=1` (inside `evaluate()`), and run a final thorough pass with `beams=5`.

---

## 5) Test on the TweetQA test set

You can load the best checkpoint and query the model with the provided `answer()` helper.

### A) Full FT checkpoint

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from helper_functions import answer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "flan_t5_tweetqa_ckpt"  # your save_dir

tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device).eval()

question = "what website is linked in the tweet?"
tweet = "I can finally say it out loud and proud: I'm going to be on Instagram!"
print(answer(model, tokenizer, device, question, tweet))
```

### B) LoRA checkpoint

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from helper_functions import answer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
base = "google/flan-t5-small"
adapter_dir = "flan_t5_tweetqa_ckpt_lora"  # your save_dir

tokenizer = AutoTokenizer.from_pretrained(base)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base).to(device)
model = PeftModel.from_pretrained(base_model, adapter_dir).eval()

q = "what website is linked in the tweet?"
t = "I can finally say it out loud and proud: I'm going to be on Instagram!"
print(answer(model, tokenizer, device, q, t))
```

## References

* TweetQA dataset (UCSB-NLP): [https://huggingface.co/datasets/ucsbnlp/tweet_qa](https://huggingface.co/datasets/ucsbnlp/tweet_qa)
* FLAN-T5 tutorial: [https://www.datacamp.com/tutorial/flan-t5-tutorial](https://www.datacamp.com/tutorial/flan-t5-tutorial)
* LoRA (PEFT docs): [https://huggingface.co/docs/peft/main/en/conceptual_guides/lora](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
