import argparse
import json
import os
import random
from dataclasses import asdict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import GPT, GPTConfig


class CharDataset(Dataset):
    def __init__(self, data: str, stoi: dict[str, int], block_size: int):
        self.data = [stoi[ch] for ch in data]
        self.block_size = block_size

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):  # type: ignore[override]
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("data", type=str, help="Path to txt file with training data")
    parser.add_argument("--out_dir", type=str, default="ckpt", help="Directory to store checkpoint")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    text = load_data(args.data)
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    train_data = CharDataset(text, stoi, args.block_size)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=len(vocab),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )

    device = torch.device(args.device)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.lr,
        (args.beta1, args.beta2),
        args.device,
    )

    for step in range(args.max_iters):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % args.eval_interval == 0:
            print(f"Step {step} Loss {loss.item():.4f}") # type: ignore

    ckpt = {
        "model": model.state_dict(),
        "config": asdict(config),
        "stoi": stoi,
        "itos": itos,
    }
    path = os.path.join(args.out_dir, "model.pt")
    torch.save(ckpt, path)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f)
    print(f"saved checkpoint to {path}")


if __name__ == "__main__":
    main()
