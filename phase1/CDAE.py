import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Config
# =========================
TRACE_FILE = "traces/traces_augmented.txt"

EPOCHS = 300
LR = 1e-3
EMB_DIM = 32
HIDDEN = 64
BATCH = 1
VERBOSE = 1
MODE = "eval"   # "train" or "eval"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rollout settings
ROLLOUT_STEPS = 24
NUM_STARTS = 5
ACTIVE_WEIGHT = 0 #4.0
workloads = [f"{app}{i}" for app in "ABCDEF" for i in range(1, 6)]

os.makedirs("models", exist_ok=True)

# =========================
# Trace Parser (FIXED)
# =========================
def parse_file(path):

    data = []

    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]

    i = 0

    while i < len(lines):

        if lines[i].startswith("#"):

            header = lines[i]
            parts = header.split()

            app = parts[2]
            intensity = parts[-2]

            wid = f"{app}{intensity}"

            meta = lines[i+1]

            zero_prefix = int(meta.split()[0].split("=")[1])
            skip_start = int(meta.split()[1].split("=")[1])
            zero_suffix = int(meta.split()[2].split("=")[1])

            cpu = list(map(float,lines[i+2].split(":")[1].split(",")))
            cache = list(map(float,lines[i+3].split(":")[1].split(",")))
            mem = list(map(float,lines[i+4].split(":")[1].split(",")))

            trace = np.stack([cpu,cache,mem],axis=1)

            data.append({
                "wid":wid,
                "trace":trace,
                "zero_prefix":zero_prefix,
                "skip_start": skip_start,
                "zero_suffix": zero_suffix
            })

            i += 6
        else:
            i += 1

    return data

def filter_dataset(dataset,target_wid):
    return [item for item in dataset if item["wid"] == target_wid]

# =========================
# Dataset
# =========================
def build_sequences(dataset):

    seqs=[]

    for item in dataset:

        wid=item["wid"]
        trace=item["trace"]
        zp=item["zero_prefix"]
        ss=item["skip_start"]
        zs=item["zero_suffix"]

        T=len(trace)
        mask=(np.sum(trace,axis=1)>0).astype(float)
        interf=np.zeros(T)

        time=np.arange(T-zp-zs) + ss
        zp_zeros=np.zeros(zp)
        zs_zeros=np.zeros(zs)
        time=np.concatenate((zp_zeros, time, zs_zeros))
        if zp>0:
            time[zp-1]=-1

        inp=np.concatenate([
            trace[:-1],
            time[:-1,None],
            interf[:-1,None]
        ],axis=1)

        seqs.append({
            "wid":wid,
            "inp":inp,
            "t_trace":trace[1:],
            "t_mask":mask[1:],
            "meta":[zp,ss,zs]
        })

    return seqs

# =========================
# Model
# =========================
class CDAE(nn.Module):

    def __init__(self, input_dim=5):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, 32)
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=HIDDEN,
            num_layers=2,
            batch_first=True
        )
        self.trace_head = nn.Linear(HIDDEN, 3)
        self.mask_head = nn.Linear(HIDDEN, 1)

    def forward(self, x, h=None):

        x = torch.relu(self.input_proj(x))
        out, h = self.gru(x, h)
        trace = self.trace_head(out)
        mask = torch.sigmoid(self.mask_head(out))
        return trace, mask, h

# =========================
# training
# =========================
def train(target_wid):
    MODEL_PATH = "models/" + target_wid + "_gru_phase1.pth"
    IMAGE_DIR = "images/phase1/" + target_wid
    os.makedirs(IMAGE_DIR, exist_ok=True)

    dataset = parse_file(TRACE_FILE)
    filtered_dataset = filter_dataset(dataset,target_wid)
    seqs = build_sequences(filtered_dataset)

    model = CDAE().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss(reduction="none")
    bce = nn.BCELoss(reduction="none")

    best = float("inf")
    loss_hist = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        np.random.shuffle(seqs)

        for s in seqs:
            inp = s["inp"]
            y_trace = s["t_trace"]
            y_mask = s["t_mask"]
            T = len(inp)

            if T <= ROLLOUT_STEPS + 1:
                continue

            opt.zero_grad()

            seq_loss = 0.0
            rollout_count = 0

            starts = [0] + list(np.random.choice(
                range(1, T - ROLLOUT_STEPS),
                size=min(4, T - ROLLOUT_STEPS - 1),
                replace=False
            ))
            for start in starts:
                h = None
                rollout_loss = 0.0

                # warmup using real history before rollout
                for t in range(start):
                    warm_x = torch.tensor(inp[t:t+1]).float().unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        _, _, h = model(warm_x, h)

                # rollout starts from real current state
                cur_x = torch.tensor(inp[start:start+1]).float().unsqueeze(0).to(DEVICE)
                
                # autoregressive rollout
                for k in range(ROLLOUT_STEPS):

                    t = start + k
                    if t >= T:
                        break

                    target_trace = torch.tensor(y_trace[t:t+1]).float().unsqueeze(0).to(DEVICE)
                    target_mask = torch.tensor(y_mask[t:t+1]).float().view(1,1,1).to(DEVICE)
                    pred_trace, pred_mask, h = model(cur_x, h)

                    trace_l = mse(pred_trace,target_trace).mean(dim=2, keepdim=True)
                    mask_l = bce(pred_mask,target_mask)
                    step_weight = 1.0 + ACTIVE_WEIGHT * target_mask
                    rollout_loss += ((trace_l + mask_l) * step_weight).mean()

                    if t + 1 < T:
                        next_time = torch.tensor(inp[t+1:t+2, 3:4]).float().unsqueeze(0).to(DEVICE)
                        next_interf = torch.tensor(inp[t+1:t+2, 4:5]).float().unsqueeze(0).to(DEVICE)
                        next_trace = pred_trace.detach()
                        cur_x = torch.cat([next_trace,next_time,next_interf], dim=2)

                rollout_loss = rollout_loss / max(1, min(ROLLOUT_STEPS, T - start))

                seq_loss += rollout_loss
                rollout_count += 1

            seq_loss = seq_loss / rollout_count
            opt.zero_grad()
            seq_loss.backward()
            opt.step()

            total_loss += seq_loss.item()

        avg_loss = total_loss / max(1, len(seqs))
        loss_hist.append(avg_loss)

        if VERBOSE:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | loss={avg_loss:.6f}")

        # Save best model
        if avg_loss < best:
            best = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)

    # Plot loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_hist, label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 1 Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    IMAGE_DIR = "images/phase1/" + target_wid
    os.makedirs(IMAGE_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGE_DIR, "train_loss.png"))
    plt.close()

    print(f"Best training loss: {best:.6f}")

# =========================
# evaluation
# =========================
def generate(model,wid,prefix,start_t,max_len,meta):

    model.eval()
    h=None
    generated=list(prefix)

    ####################################
    # warmup
    ####################################
    for t in range(len(prefix)-1):
        time=0
        if meta[0] <= t < (max_len-meta[2]):
            time=(t-meta[0]+meta[1])
        if t==meta[0]-1:
            time=-1
        
        cur=prefix[t]
        active=np.sum(np.abs(cur)) > 1e-6

        inp=np.concatenate([
            cur,
            [time],
            [0]
        ])

        x=torch.tensor(inp).float().view(1,1,5).to(DEVICE)
        _,_,h=model(x,h)

    ####################################
    # autoregressive rollout
    ####################################
    cur_trace=prefix[-1]

    for t in range(start_t,max_len):
        time=0
        if meta[0] <= t < (max_len-meta[2]):
            time=(t-meta[0]+meta[1])
        if t==meta[0]-1:
            time=-1

        active=np.sum(np.abs(cur_trace)) > 1e-6
        inp=np.concatenate([
            cur_trace,
            [time],
            [0]
        ])

        x=torch.tensor(inp).float().view(1,1,5).to(DEVICE)
        pred_trace,ـ,h=model(x,h)
        next_trace=pred_trace[0,0].detach().cpu().numpy()
        generated.append(next_trace)
        cur_trace=next_trace

    return generated

def evaluate(target_wid):
    MODEL_PATH = "models/" + target_wid + "_gru_phase1.pth"
    dataset=parse_file(TRACE_FILE)
    filtered_dataset=filter_dataset(dataset,target_wid)
    model=CDAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    IMAGE_DIR = "images/phase1/" + target_wid
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for i,item in enumerate(filtered_dataset):

        real=item["trace"]
        wid=item["wid"]
        start=0 if len(real)>20 else 0
        generated=generate(
            model,
            wid,
            real[:start+1],
            start,
            len(real),
            [item["zero_prefix"], item["skip_start"], item["zero_suffix"]]
        )

        pred_trace = np.asarray(generated, dtype=np.float32)

        fig,axs=plt.subplots(4,1,figsize=(8,8))
        names=["cpu","cache","mem"]

        for d in range(3):
            axs[d].plot(real[:,d],label="real")
            axs[d].plot(pred_trace[:,d],label="generated")
            axs[d].set_title(names[d])
        real_mask=(np.sum(real,axis=1)>0).astype(int)
        pred_mask=(np.sum(pred_trace,axis=1)>1).astype(int)

        axs[3].plot(real_mask,label="real active")
        axs[3].plot(pred_mask,label="gen active")
        axs[3].set_title("activity")
        axs[0].legend()
        plt.tight_layout()
        plt.savefig(f"{IMAGE_DIR}/eval_{i}.png")
        plt.close()

# =========================
# main
# =========================
if __name__ == "__main__":

    if MODE == "train":
        for wid in workloads:
            print(f"Training {wid}...")
            train(wid)
    for wid in workloads:
        print(f"Evaluating {wid}...")
        evaluate(wid)