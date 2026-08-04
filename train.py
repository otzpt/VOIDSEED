import torch
import numpy as np
import os
from model import GPT
import torch.nn.functional as F

train_data = np.memmap(
    "data/tinystories/train.bin",
    dtype=np.uint16,
    mode="r"
)
val_data = np.memmap(
    "data/tinystories/val.bin",
    dtype=np.uint16,
    mode="r"
)


def get_batch(split, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
        for i in ix
    ])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model = GPT(
    d_model=384,
    n_heads=6,
    n_layer=6,
    vocab_size=50257,
    block_size=256,
).to(device)

if device == "cuda":
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    fused=(device == "cuda"),
)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

os.makedirs("checkpoints", exist_ok=True)


@torch.no_grad()
def eval_val_loss():
    model.eval()
    x, y = get_batch("val", block_size=256, batch_size=8, device=device)
    with torch.amp.autocast(
        device_type=device,
        dtype=torch.float16,
        enabled=(device == "cuda"),
    ):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, 50257),
            y.reshape(-1),
        )
    model.train()
    return loss.item()


for step in range(2000):
    x, y = get_batch(
        "train",
        block_size=256,
        batch_size=8,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=device,
        dtype=torch.float16,
        enabled=(device == "cuda"),
    ):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, 50257),
            y.reshape(-1),
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
        val_loss = eval_val_loss()
        print(f"step {step}: train loss {loss.item():.4f}  val loss {val_loss:.4f}")

    if step > 0 and step % 1000 == 0:
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, f"checkpoints/{step}.pt")
