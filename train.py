import torch
import numpy as np
import os
from model import GPT
import torch.nn.functional as F

train_data = np.memmap(
    "data/security/train.bin",
    dtype=np.uint16,
    mode="r"
)
val_data = np.memmap(
    "data/security/val.bin",
    dtype=np.uint16,
    mode="r"
)

warmup_steps = 2000
max_steps = 976562

def get_lr(step):
    if step < warmup_steps:
        return 3e-4 * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 3e-4 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

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
    d_model=768,
    n_heads=12,
    n_layer=12,
    vocab_size=50257,
    block_size=256,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    fused=(device == "cuda"),
)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

start_step = 0

checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".pt")] if os.path.exists("checkpoints") else []
if checkpoint_files:
    latest = max (checkpoint_files, key = lambda f: int(f.replace(".pt", "")))
    checkpoint = torch.load(f"checkpoints/{latest}", map_location = device)

    state_dict = checkpoint["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"] + 1
    print(f"resumed from checkpoint at step {start_step}")


if device == "cuda":
    model = torch.compile(model)

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
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, 50257),
            y.reshape(-1),
        )
    model.train()
    return loss.item()


for step in range(start_step, max_steps):
    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
        logits, _ = model(x)
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

        checkpoint_files = sorted(
            [f for f in os.listdir("checkpoints") if f.endswith(".pt")],
            key = lambda f: int(f.replace(".pt", ""))
        )

        if len(checkpoint_files) > 5:
            for old_file in checkpoint_files[:-5]:
                os.remove(f"checkpoints/{old_file}")
