import ast
import math
import re
import string
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase


# ---------------------------
# Dataset + Collate
# ---------------------------
class TweetQADataset(Dataset):
    """
    Expects a DataFrame with columns: Question, Answer, Tweet, qid
    Builds a prompt like:
        "Question: {question}\nTweet: {tweet}\nAnswer:"
    Target is the first element of Answer if it's a list-string; else Answer as-is.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_tmpl: str = "Question: {question}\nTweet: {tweet}\nAnswer:",
        use_lowercase_answer: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.prompt_tmpl = prompt_tmpl
        self.use_lowercase_answer = use_lowercase_answer

        # Basic sanity
        missing = {"Question", "Answer", "Tweet"} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _parse_first_answer(ans_cell):
        # Handles strings like "[instagram]" or '["foo","bar"]' or a plain "instagram"
        if isinstance(ans_cell, list):
            return ans_cell[0] if ans_cell else ""
        if isinstance(ans_cell, str):
            s = ans_cell.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list) and parsed:
                        return str(parsed[0])
                except Exception:
                    pass
            return s
        # Fallback
        if ans_cell is None or (isinstance(ans_cell, float) and math.isnan(ans_cell)):
            return ""
        return str(ans_cell)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["Question"])
        tweet = str(row["Tweet"])
        target = self._parse_first_answer(row["Answer"])

        if self.use_lowercase_answer:
            target = target.lower().strip()

        prompt = self.prompt_tmpl.format(question=question, tweet=tweet)

        return {
            "prompt": prompt,
            "target": target,
            "qid": row.get("qid", None),
            "tweet": tweet,
        }


def build_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_input_len: int = 256,
    max_target_len: int = 32,
):
    """
    Tokenizes prompts and targets; pads; sets labels; also passes through qid/tweet as lists.
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]
        qids    = [b.get("qid") for b in batch]
        tweets  = [b.get("tweet") for b in batch]

        model_inputs = tokenizer(
            prompts,
            max_length=max_input_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Newer tokenizers allow: tokenizer(text_target=targets, ...)
        # For backward compatibility keep as_target_tokenizer
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

        # Replace pad token IDs in labels with -100 so they’re ignored by the loss
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        # Carry metadata (CPU lists) for error analysis
        model_inputs["qid"]   = qids
        model_inputs["tweet"] = tweets
        model_inputs["gold_text"] = targets
        return model_inputs

    return collate_fn


# ---------------------------
# Metrics + utilities
# ---------------------------
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, truth: str) -> float:
    return float(_norm(pred) == _norm(truth))


def f1_score(pred: str, truth: str) -> float:
    p = _norm(pred).split()
    t = _norm(truth).split()
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    from collections import Counter
    cp, ct = Counter(p), Counter(t)
    num_same = sum(min(cp[w], ct[w]) for w in cp)
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec = num_same / len(t)
    return 2 * prec * rec / (prec + rec)


def tag_tweet(t: str) -> Dict[str, int]:
    return {
        "has_hashtag": int("#" in t),
        "has_handle": int("@" in t),
        "has_url": int(bool(re.search(r"https?://", t))),
        "has_number": int(bool(re.search(r"\d", t))),
    }


# ---------------------------
# Semantic similarity
# ---------------------------
def semantic_sim_batch(
    sem_model,  # SentenceTransformer
    pred_texts: List[str],
    gold_texts: List[str],
) -> List[float]:
    """
    Sentence-embeddings cosine similarity in [0,1].
    Assumes sem_model is already placed on the right device and in eval() mode.
    """
    emb_pred = sem_model.encode(
        pred_texts, convert_to_tensor=True, normalize_embeddings=True,
        batch_size=64, show_progress_bar=False
    )
    emb_gold = sem_model.encode(
        gold_texts, convert_to_tensor=True, normalize_embeddings=True,
        batch_size=64, show_progress_bar=False
    )
    sim = (emb_pred * emb_gold).sum(dim=1)  # cosine since normalized
    return sim.clamp(min=-1.0, max=1.0).tolist()


# ---------------------------
# Evaluation + answer
# ---------------------------
@torch.no_grad()
def evaluate(
    model,
    val_loader,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    sem_model=None,  # optional SentenceTransformer for semantic similarity
    gen_max_new_tokens: int = 12,
    beams: int = 5,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs validation; returns loss/EM/F1/avg_sem_sim + per-example rows DataFrame.
    """
    model.eval()
    losses, em_list, f1_list, sem_sim_list = [], [], [], []
    rows = []

    for bi, batch in enumerate(tqdm(val_loader, leave=False)):
        if max_batches is not None and bi >= max_batches:
            break

        # Move only tensors to device
        tensor_keys = ["input_ids", "attention_mask", "labels"]
        tb = {k: v.to(device) for k, v in batch.items() if k in tensor_keys}

        # forward (new AMP API)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            out = model(**tb)
        losses.append(float(out.loss.detach()))

        # generate
        gen_ids = model.generate(
            input_ids=tb["input_ids"],
            attention_mask=tb["attention_mask"],
            max_new_tokens=gen_max_new_tokens,
            num_beams=beams,
            do_sample=False,
            length_penalty=0.0,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # decode gold labels
        gold_ids = tb["labels"].clone()
        gold_ids[gold_ids == -100] = tokenizer.pad_token_id
        golds = tokenizer.batch_decode(gold_ids, skip_special_tokens=True)

        # metadata stays on CPU as lists
        qids   = batch.get("qid",   [None] * len(preds))
        tweets = batch.get("tweet", [""]    * len(preds))

        # semantic similarity per batch (optional)
        if sem_model is not None:
            sims = semantic_sim_batch(sem_model, preds, golds)
            sem_sim_list.extend(sims)
        else:
            sims = [0.0] * len(preds)

        # scoring + rows
        for qid, pred, gold, tweet, sim in zip(qids, preds, golds, tweets, sims):
            em_i = exact_match(pred, gold)
            f1_i = f1_score(pred, gold)
            em_list.append(em_i)
            f1_list.append(f1_i)
            rows.append({
                "qid": qid,
                "pred": pred,
                "gold": gold,
                "f1": f1_i,
                "em": em_i,
                "sem_sim": sim,
                "alen": len(gold.split()),
                **tag_tweet(tweet),
            })

    df_err = pd.DataFrame(rows)
    model.train()  # restore training mode
    return {
        "val_loss": sum(losses) / len(losses) if losses else 0.0,
        "em":       sum(em_list) / len(em_list) if em_list else 0.0,
        "f1":       sum(f1_list) / len(f1_list) if f1_list else 0.0,
        "avg_sem_sim": sum(sem_sim_list) / len(sem_sim_list) if sem_sim_list else 0.0,
        "errors":   df_err,
    }


@torch.no_grad()
def answer(
    model,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    question: str,
    tweet: str,
    max_new_tokens: int = 12,
) -> str:
    prompt = f"Question: {question}\nTweet: {tweet}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt")
    # BatchEncoding has .to(); this moves tensors only
    enc = enc.to(device)
    model.eval()
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        do_sample=False,
        length_penalty=0.0,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True)
