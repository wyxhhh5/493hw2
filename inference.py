import argparse
import torch

from model import GPT, GPTConfig


def sample(model: GPT, idx: torch.Tensor, length: int, stoi: dict[str, int], itos: dict[int, str]) -> str:
    device = idx.device
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            idx_cond = idx[-model.config.block_size:]
            logits = model(idx_cond.unsqueeze(0))[0, -1]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=0)
    text = "".join(itos[int(i)] for i in idx.tolist())
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint pt file")
    parser.add_argument("--prompt", type=str, default="", help="Starting text")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()} if isinstance(ckpt["itos"], dict) else ckpt["itos"]

    if len(args.prompt) == 0:
        idx = torch.tensor([stoi[next(iter(stoi))]], dtype=torch.long, device=args.device)
    else:
        idx = torch.tensor([stoi[ch] for ch in args.prompt], dtype=torch.long, device=args.device)

    result = sample(model, idx, args.steps, stoi, itos)
    print(result)


if __name__ == "__main__":
    main()
