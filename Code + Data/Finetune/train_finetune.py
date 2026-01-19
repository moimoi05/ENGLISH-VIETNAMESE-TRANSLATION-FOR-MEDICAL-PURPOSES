# train_finetune.py - FINETUNE 300K, A100, AMP BF16

import os
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from datasets import load_from_disk
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# -------------------------------------------------
# FIX IMPORT PATH
# -------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import build_transformer
from med_dataset import MedTranslationDataset
from config_finetune import config

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(config["save_dir"], exist_ok=True)

# -------------------------------------------------
# TENSORBOARD
# -------------------------------------------------
writer = SummaryWriter(log_dir="runs/tmodel_finetune")
global_step = 0

# -------------------------------------------------
# TOKENIZERS & SPECIAL IDS
# -------------------------------------------------
tokenizer_src = Tokenizer.from_file("../tokenizer_en.json")
tokenizer_tgt = Tokenizer.from_file("../tokenizer_vi.json")

PAD_ID = tokenizer_tgt.token_to_id("[PAD]")
EOS_ID = tokenizer_tgt.token_to_id("[EOS]")
SOS_ID = tokenizer_tgt.token_to_id("[SOS]")

# -------------------------------------------------
# COLLATE FN: PAD THEO BATCH
# -------------------------------------------------
def collate_fn(batch):
    encoder_input_list = [b["encoder_input"] for b in batch]
    decoder_input_list = [b["decoder_input"] for b in batch]
    label_list = [b["label"] for b in batch]
    src_text_list = [b["src_text"] for b in batch]
    tgt_text_list = [b["tgt_text"] for b in batch]

    encoder_input = pad_sequence(
        encoder_input_list, batch_first=True, padding_value=PAD_ID
    )
    decoder_input = pad_sequence(
        decoder_input_list, batch_first=True, padding_value=PAD_ID
    )
    label = pad_sequence(
        label_list, batch_first=True, padding_value=PAD_ID
    )

    # (B, 1, 1, S)
    encoder_mask = (encoder_input != PAD_ID).unsqueeze(1).unsqueeze(2).int()
    # (B, 1, T, T)
    size = decoder_input.size(1)
    causal = torch.tril(
        torch.ones(1, size, size, dtype=torch.int, device=encoder_input.device)
    )
    decoder_mask = (
        (decoder_input != PAD_ID).unsqueeze(1).unsqueeze(2).int() & causal
    )

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "label": label,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "src_text": src_text_list,
        "tgt_text": tgt_text_list,
    }

# -------------------------------------------------
# BEAM SEARCH
# -------------------------------------------------
def beam_search_translate(
    model,
    sentence,
    tokenizer_src,
    tokenizer_tgt,
    beam_size=5,
    max_len=100,
    length_penalty=0.8,
    repetition_penalty=1.4,
):
    model.eval()
    device = next(model.parameters()).device
    sos, eos = SOS_ID, EOS_ID

    src_ids = tokenizer_src.encode(sentence).ids
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=device)

    with torch.no_grad():
        enc_out = model.encode(src, src_mask)

    beams = [([sos], 0.0)]
    for _ in range(max_len):
        candidates = []
        for tokens, score in beams:
            if tokens[-1] == eos:
                candidates.append((tokens, score))
                continue

            tgt = torch.tensor(tokens, device=device).unsqueeze(0)
            tgt_mask = torch.tril(
                torch.ones(1, tgt.size(1), tgt.size(1), device=device)
            ).bool()

            with torch.no_grad():
                dec_out = model.decode(enc_out, src_mask, tgt, tgt_mask)
                logits = model.project(dec_out[:, -1]).squeeze(0)
                log_probs = F.log_softmax(logits, dim=-1)

            for t in set(tokens):
                log_probs[t] /= repetition_penalty

            topk = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                next_tok = topk.indices[i].item()
                next_score = score + topk.values[i].item()
                candidates.append((tokens + [next_tok], next_score))

        beams = sorted(
            candidates,
            key=lambda x: x[1] / (len(x[0]) ** length_penalty),
            reverse=True,
        )[:beam_size]

        if all(b[0][-1] == eos for b in beams):
            break

    best = beams[0][0]
    if eos in best:
        best = best[1:best.index(eos)]
    else:
        best = best[1:]

    return tokenizer_tgt.decode(best)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
train_hf = load_from_disk(config["train_path"])
val_hf = load_from_disk(config["val_path"])

train_dataset = MedTranslationDataset(
    train_hf, tokenizer_src, tokenizer_tgt, config["seq_len"]
)
val_dataset = MedTranslationDataset(
    val_hf, tokenizer_src, tokenizer_tgt, config["seq_len"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# -------------------------------------------------
# MODEL: BUILD → LOAD CKPT → COMPILE
# -------------------------------------------------
model = build_transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    config["seq_len"],
    config["d_model"],
).to(device)

print("Loading pretrained checkpoint:", config["pretrained_ckpt"])
ckpt = torch.load(config["pretrained_ckpt"], map_location=device)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

# xử lý prefix _orig_mod. trong checkpoint cũ
clean_state = {}
for k, v in state_dict.items():
    new_k = k.replace("_orig_mod.", "")
    clean_state[new_k] = v

model.load_state_dict(clean_state)
print("Loaded pretrained model weights")

print("Compiling model...")
model = torch.compile(model)

# -------------------------------------------------
# ENCODER FREEZE STRATEGY
# -------------------------------------------------
def set_encoder_trainable(model, epoch: int):
    if epoch == 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Epoch 0: Encoder FULLY frozen")
    elif epoch < 6:
        for i, layer in enumerate(model.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = (i >= len(model.encoder.layers) - 4)
        print(f"Epoch {epoch}: Encoder TOP-4 layers unfrozen")
    else:
        for p in model.encoder.parameters():
            p.requires_grad = True
        print(f"Epoch {epoch}: FULL encoder unfrozen")

model.train()

# -------------------------------------------------
# OPTIMIZER & LOSS
# -------------------------------------------------
optimizer = torch.optim.AdamW(
    [
        {"params": model.decoder.parameters(), "lr": 1e-4},
        {"params": model.encoder.parameters(), "lr": 2e-5},
    ],
    weight_decay=config["weight_decay"],
)

vocab_size = tokenizer_tgt.get_vocab_size()
weights = torch.ones(vocab_size, device=device)
weights[EOS_ID] = 3.0
weights[PAD_ID] = 0.0

criterion = torch.nn.CrossEntropyLoss(
    weight=weights,
    ignore_index=PAD_ID,
    label_smoothing=config.get("label_smoothing", 0.1),
)

# -------------------------------------------------
# EARLY STOPPING
# -------------------------------------------------
patience = 4
no_improve = 0
best_val_loss = float("inf")

# -------------------------------------------------
# TRAIN LOOP (AMP BF16 + GRAD ACCUMULATION)
# -------------------------------------------------
accumulation_steps = 2  # batch 128 -> effective 256

for epoch in range(config["num_epochs"]):
    set_encoder_trainable(model, epoch)
    model.train()

    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

    for i, batch in enumerate(progress):
        src = batch["encoder_input"].to(device)
        tgt_in = batch["decoder_input"].to(device)
        tgt_out = batch["label"].to(device)
        src_mask = batch["encoder_mask"].to(device)
        tgt_mask = batch["decoder_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)
            logits = model.project(dec_out)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1),
            )
            loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        writer.add_scalar("train loss", loss.item(), global_step)
        global_step += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            src = batch["encoder_input"].to(device)
            tgt_in = batch["decoder_input"].to(device)
            tgt_out = batch["label"].to(device)
            src_mask = batch["encoder_mask"].to(device)
            tgt_mask = batch["decoder_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc_out = model.encode(src, src_mask)
                dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)
                logits = model.project(dec_out)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    tgt_out.view(-1),
                )
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("val loss", avg_val_loss, epoch)

    print(
        f"\nEpoch {epoch} DONE | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}\n"
    )

    # EARLY STOPPING
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # QUALITATIVE EVAL
    sample_ids = random.sample(range(min(10, len(val_hf))), 2)
    for idx in sample_ids:
        src_text = val_hf[idx]["Eng"]
        tgt_text = val_hf[idx]["Vie"]
        pred = beam_search_translate(model, src_text, tokenizer_src, tokenizer_tgt)
        print("-" * 80)
        print("Source  :", src_text)
        print("Expected:", tgt_text)
        print("Predicted:", pred)
    print("-" * 80)

    # SAVE BEST
    if avg_val_loss <= best_val_loss:
        save_path = os.path.join(config["save_dir"], "finetune_best.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
            },
            save_path,
        )
        print("Saved BEST model")

    # SAVE LATEST
    latest_path = os.path.join(config["save_dir"], "finetune_latest.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
        },
        latest_path,
    )
    print(f"Saved latest epoch {epoch}")

writer.close()
print("Training finished.")
