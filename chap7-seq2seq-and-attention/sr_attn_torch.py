import argparse
import random
import string
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


VOCAB_SIZE = 27  # 0 is <s>, 1..26 => A..Z


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_string(length: int) -> str:
    letters = string.ascii_uppercase
    return "".join(random.choice(letters) for _ in range(length))


def get_batch(batch_size: int, length: int, device: torch.device) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    batched_examples = [random_string(length) for _ in range(batch_size)]
    enc_x = [[ord(ch) - ord("A") + 1 for ch in seq] for seq in batched_examples]
    y = [list(reversed(item)) for item in enc_x]
    dec_x = [[0] + item[:-1] for item in y]
    return (
        batched_examples,
        torch.tensor(enc_x, dtype=torch.long, device=device),
        torch.tensor(dec_x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.GRU(emb_dim, hidden, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hidden, batch_first=True)
        self.attn_proj = nn.Linear(hidden, hidden, bias=False)  # bilinear attention
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, enc_ids: torch.Tensor, dec_ids: torch.Tensor) -> torch.Tensor:
        enc_emb = self.embedding(enc_ids)
        enc_out, h = self.encoder(enc_emb)  # enc_out: [b, src_len, h]

        dec_emb = self.embedding(dec_ids)
        dec_out, _ = self.decoder(dec_emb, h)  # [b, tgt_len, h]

        scores = torch.matmul(self.attn_proj(dec_out), enc_out.transpose(1, 2))  # [b, tgt_len, src_len]
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, enc_out)  # [b, tgt_len, h]

        logits = self.output(dec_out + context)
        return logits

    @torch.no_grad()
    def encode(self, enc_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_emb = self.embedding(enc_ids)
        enc_out, h = self.encoder(enc_emb)
        return enc_out, h

    @torch.no_grad()
    def get_next_token(self, token: torch.Tensor, state: torch.Tensor, enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token).unsqueeze(1)  # [b, 1, emb]
        dec_out, state = self.decoder(emb, state)  # [b, 1, h]
        dec_h = dec_out.squeeze(1)

        scores = torch.bmm(self.attn_proj(dec_h).unsqueeze(1), enc_out.transpose(1, 2))  # [b, 1, src_len]
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, enc_out).squeeze(1)  # [b, h]

        logits = self.output(dec_h + context)
        next_token = torch.argmax(logits, dim=-1)
        return next_token, state


def train(model: Seq2SeqWithAttention, steps: int, seqlen: int, batch_size: int, lr: float, device: torch.device) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for step in range(steps):
        _, enc_x, dec_x, y = get_batch(batch_size, seqlen, device)
        logits = model(enc_x, dec_x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % max(1, steps // 6) == 0:
            print(f"step {step}: loss {loss.item():.4f}")


@torch.no_grad()
def sequence_reversal(model: Seq2SeqWithAttention, batch_size: int, length: int, device: torch.device) -> Tuple[List[str], List[str]]:
    model.eval()
    examples, enc_x, _, _ = get_batch(batch_size, length, device)
    enc_out, state = model.encode(enc_x)

    cur_token = torch.zeros(batch_size, dtype=torch.long, device=device)
    outs = []
    for _ in range(length):
        cur_token, state = model.get_next_token(cur_token, state, enc_out)
        outs.append(cur_token.unsqueeze(1))
    pred_ids = torch.cat(outs, dim=1).cpu().tolist()
    preds = ["".join(chr(i + ord("A") - 1) for i in row) for row in pred_ids]
    return preds, examples


def is_reverse(seq: str, rev_seq: str) -> bool:
    return seq == rev_seq[::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"use gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("use cpu")
    model = Seq2SeqWithAttention().to(device)

    train(model, args.steps, args.seq_len, args.batch_size, args.lr, device)
    preds, src = sequence_reversal(model, batch_size=16, length=args.seq_len, device=device)

    checks = [is_reverse(s, p) for p, s in zip(preds, src)]
    print("sample (pred, src):")
    for item in list(zip(preds, src))[:5]:
        print(item)
    print(f"reverse accuracy on sample batch: {sum(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
