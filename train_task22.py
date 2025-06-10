import argparse
import json
import os
import random
from typing import List, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from model import GPT, GPTConfig


class ArithmeticDataset(Dataset):
    def __init__(self, path: str, stoi: dict, block_size: int, operators: List[str], prime: int | None = None, pad_token: str = "<pad>"):
        self.stoi = stoi
        self.block_size = block_size
        self.pad_idx = stoi[pad_token]
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, op, b, _, c = line.split()
                if op not in operators:
                    continue
                a, b, c = int(a), int(b), int(c)
                if prime is not None:
                    if op == "+" and (a + b) % prime != c:
                        continue
                    if op == "-" and (a - b) % prime != c:
                        continue
                line = line + "\n"
                tokens = [stoi[ch] for ch in line]
                tokens = tokens[: block_size]
                pad_len = block_size - len(tokens)
                tokens += [self.pad_idx] * pad_len
                x = torch.tensor(tokens[:-1], dtype=torch.long)
                y = torch.tensor(tokens[1:], dtype=torch.long)
                out_start = line.index("=") + 2
                target = y.clone()
                target[: out_start - 1] = -100
                self.data.append((x, target))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(paths: List[str]) -> Tuple[List[str], dict, dict]:
    chars = set()
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            chars.update(set(f.read()))
    chars.add("\n")
    vocab = ["<pad>"] + sorted(chars)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return vocab, stoi, itos


def evaluate(model: GPT, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(-1)
            mask = y != -100
            total_correct += ((preds == y) & mask).sum().item()
            total_count += mask.sum().item()
    return total_loss / len(loader.dataset), total_correct / total_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model for addition/subtraction tasks")
    parser.add_argument("data_dir", type=str, help="Directory with train.txt, val.txt, test.txt")
    parser.add_argument("--prime", type=int, choices=[97, 113], required=True)
    parser.add_argument("--operator", type=str, choices=["+", "-"], default="+")
    parser.add_argument("--out_dir", type=str, default="ckpt_task22")
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = [os.path.join(args.data_dir, f) for f in ["train.txt", "val.txt", "test.txt"]]
    vocab, stoi, itos = build_vocab(paths)
    line_lengths = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            line_lengths.extend(len(line.strip()) + 1 for line in f)
    block_size = max(line_lengths)

    train_dataset = ArithmeticDataset(paths[0], stoi, block_size, [args.operator], args.prime)
    val_dataset = ArithmeticDataset(paths[1], stoi, block_size, [args.operator], args.prime)
    test_dataset = ArithmeticDataset(paths[2], stoi, block_size, [args.operator], args.prime)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    config = GPTConfig(
        block_size=block_size - 1,
        vocab_size=len(vocab),
        n_layer=args.n_layer,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
    )
    device = torch.device(args.device)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), args.device)

    data_iter = iter(train_loader)
    for step in range(1, args.max_iters + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % args.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"step {step} train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    ckpt = {
        "model": model.state_dict(),
        "config": config.__dict__,
        "stoi": stoi,
        "itos": itos,
    }
    path = os.path.join(args.out_dir, f"model_p{args.prime}_{args.operator}.pt")
    torch.save(ckpt, path)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f)
    print(f"saved checkpoint to {path}")


if __name__ == "__main__":
    main()
