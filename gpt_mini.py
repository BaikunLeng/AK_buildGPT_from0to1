import os, math, argparse, requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
import wandb

print("[RUN FILE]", __file__)

# ------------------ 超参 ------------------
batch_size = 16
block_size = 32
max_iters = int(os.getenv("MAX_ITERS", 5000))
eval_interval = 100
learning_rate = 1e-3
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
torch.manual_seed(1337)

# ------------------ CLI -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["tinyshakespeare", "wikitext2", "ptb"], default="tinyshakespeare")
args = parser.parse_args()
print("Using device:", device, "| Dataset:", args.dataset)

# ---------------- 数据加载 -----------------
def load_tinyshakespeare() -> Tuple[str, str]:
    if not os.path.exists("input.txt"):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write(requests.get(url, timeout=30).text)
    with open("input.txt", "r", encoding="utf-8") as f:
        full = f.read()
    n = int(0.9 * len(full))
    return full[:n], full[n:]

def load_wikitext2() -> Tuple[str, str]:
    # 仍然通过 HF Hub 加载（这个数据集是 OK 的）
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n\n".join(ds["train"]["text"])
    val_text   = "\n\n".join(ds["validation"]["text"])
    return train_text, val_text

def load_ptb() -> Tuple[str, str]:
    # 只用原始 URL，完全不再调用 load_dataset
    print("[PTB] Loaded from raw URLs (no HF datasets).")
    urls = {
        "train": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
        "valid": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
    }
    os.makedirs("data_ptb", exist_ok=True)
    for split, url in urls.items():
        p = f"data_ptb/ptb.{split}.txt"
        if not os.path.exists(p):
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(p, "w", encoding="utf-8") as f:
                f.write(r.text)
    with open("data_ptb/ptb.train.txt", "r", encoding="utf-8") as f:
        train_text = f.read()
    with open("data_ptb/ptb.valid.txt", "r", encoding="utf-8") as f:
        val_text = f.read()
    return train_text, val_text

if args.dataset == "tinyshakespeare":
    train_text, val_text = load_tinyshakespeare()
elif args.dataset == "wikitext2":
    train_text, val_text = load_wikitext2()
else:  # ptb
    train_text, val_text = load_ptb()

print(f"Loaded {args.dataset}: train chars={len(train_text):,}, val chars={len(val_text):,}")

# ------------- 构建词表/编码 --------------
chars = sorted(list(set(train_text + val_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data   = torch.tensor(encode(val_text),   dtype=torch.long)
print("vocab_size =", vocab_size, "| train tokens =", len(train_data), "| val tokens =", len(val_data))

# --------------- W&B ----------------------
wandb.login()
run = wandb.init(
    project="mini-gpt-benchmark",
    name=f"{args.dataset}-emb{n_embd}-L{n_layer}-H{n_head}-blk{block_size}-bs{batch_size}-lr{learning_rate}-seed1337",
    group=args.dataset,
    config=dict(
        dataset=args.dataset, device=device,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        block_size=block_size, batch_size=batch_size,
        learning_rate=learning_rate, dropout=dropout,
        max_iters=max_iters, eval_interval=eval_interval,
        vocab_size=vocab_size,
        train_tokens=len(train_data), val_tokens=len(val_data),
        seed=1337,
    ),
)

# ------------- dataloader -----------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ------------- 模型定义 -------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        wei = q @ k.transpose(-2,-1) * (q.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1); wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ------------- 训练循环 -------------------
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        bpc = float(losses['val']) / math.log(2.0)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val BPC {bpc:.3f}")
        wandb.log({
            "iter": iter,
            "train_loss": float(losses['train']),
            "val_loss": float(losses['val']),
            "val_bpc": bpc,
            "lr": learning_rate,
        })

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ------------- 采样 + 结束 ----------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
sample = ''.join([decode(model.generate(context, max_new_tokens=400)[0].tolist())])
print(sample)
wandb.log({"sample_text": wandb.Html(f"<pre>{sample}</pre>")})
wandb.finish()
