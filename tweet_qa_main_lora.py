import os, math, time, re, ast, string
from peft import LoraConfig, TaskType, get_peft_model
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer

from helper_function import (
    TweetQADataset, build_collate_fn,
    evaluate, answer
)
# ---------------- CONFIG ----------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--save_dir", default="flan_t5_tweetqa_ckpt_lora")
    return parser.parse_args()


# ---------------- TRAIN FUNCTION ----------------
def train(model, train_loader, val_loader, tokenizer, sem_model, device, args):
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    num_training_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(0.06 * num_training_steps)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_params, lr=args.lr)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_losses, val_losses, val_f1s, val_avg_sem_sims = [], [], [], []
    best_f1 = -1.0
    os.makedirs(args.save_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        step_count = 0
        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader))
        running = 0.0

        model.train()
        for step, batch in pbar:
            tensor_keys = ["input_ids", "attention_mask", "labels"]
            batch_t = {k: v.to(device) for k, v in batch.items() if k in tensor_keys}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                out = model(**batch_t)
                loss = out.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            step_count += 1
            running += loss.detach().float().item()
            if step % 50 == 0:
                avg_loss = running / 50
                train_losses.append(avg_loss)
                pbar.set_description(f"epoch {epoch} | step {step} | loss {avg_loss:.4f}")
                running = 0.0

        epoch_time = time.time() - epoch_t0
        steps_per_sec = step_count / epoch_time
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0
        print(f"[epoch {epoch}] time={epoch_time/60:.2f} min | steps/sec={steps_per_sec:.2f} | peak VRAM={peak_vram_mb:.0f} MB")

        # ---- Evaluate ----
        metrics = evaluate(model, val_loader, device, tokenizer)
        val_losses.append(metrics['val_loss'])
        val_f1s.append(metrics['f1'])
        val_avg_sem_sims.append(metrics['avg_sem_sim'])
        print(f"[epoch {epoch}] val_loss={metrics['val_loss']:.4f} F1={metrics['f1']:.4f}, avg_sem_sim={metrics['avg_sem_sim']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            df_err = metrics.get("errors", None)
            if df_err is not None:
                df_err.to_csv(os.path.join(args.save_dir, f"errors_epoch{epoch}.csv"), index=False)
            print(f"✅ Saved best to {args.save_dir} (F1={best_f1:.4f})")

    # ---- Plotting ----
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train loss (per 50 steps)")
    plt.plot([i*len(train_loader)//50 for i in range(1, len(val_losses)+1)], val_losses, label="Val loss (per epoch)")
    plt.xlabel("Steps"); plt.ylabel("Loss")
    plt.title("Training / Validation Loss"); plt.legend()
    save_path = os.path.join(args.save_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(val_f1s)+1), val_f1s, marker='o', label="Val F1 per epoch")
    plt.xlabel("Epoch"); plt.ylabel("F1")
    plt.title("Validation F1 progression"); plt.legend()
    save_path = os.path.join(args.save_dir, "val_f1_curve.png")
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(val_avg_sem_sims)+1), val_avg_sem_sims, marker='o', label="Val Avg Semantic Similarity per epoch")
    plt.xlabel("Epoch"); plt.ylabel("Avg Semantic Similarity")
    plt.title("Validation Avg Semantic Similarity progression"); plt.legend()
    save_path = os.path.join(args.save_dir, "val_avg_sem_sim_curve.png")
    plt.savefig(save_path)
    plt.close()


# ---------------- MAIN ----------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    ds = load_dataset("ucsbnlp/tweet_qa")
    train_df = ds["train"].to_pandas()
    val_df   = ds["validation"].to_pandas()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,                 # rank; try 4–16
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q","v"],  # T5 attention proj names
    bias="none"
    )
    
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    # Semantic similarity model
    sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sem_model = sem_model.to(device if torch.cuda.is_available() else "cpu").eval()

    # Dataloaders
    collate_fn = build_collate_fn(tokenizer, max_input_len=256, max_target_len=32)
    train_dataset = TweetQADataset(train_df)
    val_dataset   = TweetQADataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Train
    train(model, train_loader, val_loader, tokenizer, sem_model, device, args)


if __name__ == "__main__":
    main()
