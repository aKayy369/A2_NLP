import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request
import re
import os

# --------------------
# App Setup
# --------------------
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved", "best-val-lstm_lm.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "saved", "vocab.pt")

# --------------------
# Tokenizer (same as training)
# --------------------
def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text)

# --------------------
# Load vocab
# --------------------
vocab_data = torch.load(VOCAB_PATH, map_location="cpu")
stoi = vocab_data["stoi"]
itos = vocab_data["itos"]

UNK_IDX = stoi["<unk>"]
EOS_IDX = stoi["<eos>"]
vocab_size = len(itos)

# --------------------
# Model Definition (MUST MATCH TRAINING)
# --------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return h, c

    def forward(self, x, hidden):
        x = self.dropout(self.embedding(x))
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

# --------------------
# Load model
# --------------------
model = LSTMLanguageModel(
    vocab_size=vocab_size,
    emb_dim=1024,
    hid_dim=1024,
    num_layers=2,
    dropout=0.65
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --------------------
# Text Generation (CLEAN OUTPUT)
# --------------------
@torch.no_grad()
def generate_text(prompt, max_len=80, temperature=0.8):
    tokens = tokenize(prompt)
    ids = [stoi.get(t, UNK_IDX) for t in tokens]
    if not ids:
        ids = [UNK_IDX]

    idx = torch.tensor(ids, device=device).unsqueeze(0)
    hidden = model.init_hidden(1)

    for _ in range(max_len):
        logits, hidden = model(idx, hidden)
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id == EOS_IDX:
            break

        idx = torch.cat(
            [idx, torch.tensor([[next_id]], device=device)], dim=1
        )

    # ---- CLEAN SPECIAL TOKENS FOR UI ----
    tokens = [itos[i] for i in idx[0].tolist()]
    tokens = [t for t in tokens if not t.startswith("<")]

    return " ".join(tokens)

# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    prompt = ""
    temperature = 0.8

    if request.method == "POST":
        prompt = request.form["prompt"]
        temperature = float(request.form["temperature"])
        output = generate_text(prompt, temperature=temperature)

    return render_template(
        "index.html",
        prompt=prompt,
        output=output,
        temperature=temperature
    )

if __name__ == "__main__":
    app.run(debug=True)
