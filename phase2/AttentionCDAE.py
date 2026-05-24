import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_COLOCATED = 6
HIDDEN = 64
ROLLOUT_STEPS = 24
EPOCHS = 60
TRAIN_SEGMENTS_PER_DATASET = 5  # None keeps the old len(wid) behavior

LR_CDAE = 1e-4
LR_AGG  = 5e-4
MODE = "train"   # "train" or "eval"

MODEL_DIR = "models"
TRACE_FILE = "traces/colocated_traces.txt"
SAVE_PATH = MODEL_DIR+"/phase2_best.pth"
IMAGES_DIR = "images/phase2"

all_workloads = [f"{app}{i}" for app in "ABCDEF" for i in range(1, 6)]
wid_map = {w: i for i, w in enumerate(all_workloads)}

# =========================================================
# Parse
# =========================================================
def build_input(data, w, t):
    # Active check: sum of abs values > 1e-6 (from CDAE.py)
    trace_vec = np.array([data[w]["cpu"][t], data[w]["cache"][t], data[w]["mem"][t]])
    active = 1.0 if np.sum(np.abs(trace_vec)) > 0.5 else 0.0
    
    offset = data[w]["offset"]

    time = 0
    if active:
        time=t-offset
    elif t==offset-1:
        time=-1
    
    return torch.tensor([
        data[w]["cpu"][t],
        data[w]["cache"][t],
        data[w]["mem"][t],
        time
    ]).float().view(1,1,4).to(DEVICE)

def build_inputs_and_targets(data, wid):
    inputs = {}
    targets = {}

    for w in wid:
        cpu = torch.tensor(data[w]["cpu"], dtype=torch.float32, device=DEVICE)
        cache = torch.tensor(data[w]["cache"], dtype=torch.float32, device=DEVICE)
        mem = torch.tensor(data[w]["mem"], dtype=torch.float32, device=DEVICE)
        trace = torch.stack([cpu, cache, mem], dim=1)

        active = trace.abs().sum(dim=1) > 0.5
        time = torch.zeros_like(cpu)
        active_idx = active.nonzero(as_tuple=True)[0]
        time[active_idx] = active_idx.float() - data[w]["offset"]
        if 0 <= data[w]["offset"] - 1 < time.numel():
            time[data[w]["offset"] - 1] = torch.where(
                active[data[w]["offset"] - 1],
                time[data[w]["offset"] - 1],
                torch.tensor(-1.0, device=DEVICE)
            )

        inputs[w] = torch.cat([trace, time.unsqueeze(1)], dim=1)
        targets[w] = trace

    return inputs, targets

def parse_trace_file(path):

    with open(path, "r") as f:
        content = f.read()

    blocks = content.split("########################")
    datasets = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        workloads = re.findall(r"[A-Z]\d", lines[0])

        data = {}
        current = None

        for line in lines[1:]:
            line = line.strip()

            if "+" in line:
                parts = line.split()
                offset = int(parts[1].replace("+",""))
                current = parts[0]
                data[current] = {"cpu": [], "cache": [], "mem": [], "offset": offset}

            elif "core cpu usage" in line:
                data[current]["cpu"] = list(map(float, line.split(":")[1].split(",")))

            elif "cache miss" in line:
                data[current]["cache"] = list(map(float, line.split(":")[1].split(",")))

            elif "mem bw" in line:
                data[current]["mem"] = list(map(float, line.split(":")[1].split(",")))

        datasets.append(data)

    return datasets

# =========================================================
# Model
# =========================================================
class CDAE(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 32)
        self.gru = nn.GRU(32, HIDDEN, num_layers=2, batch_first=True)
        self.trace_head = nn.Linear(HIDDEN, 3)
        self.mask_head = nn.Linear(HIDDEN, 1)

    def forward(self, x, h=None):
        x = torch.relu(self.input_proj(x))
        out, h = self.gru(x, h)
        trace = self.trace_head(out)
        mask = torch.sigmoid(self.mask_head(out))
        return trace, mask, h

class InterferenceAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, traces):
        # traces: list of [B,3]
        elems = [self.phi(t) for t in traces]
        summed = torch.stack(elems, dim=0).sum(dim=0)
        return self.rho(summed)

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.cdae_models = nn.ModuleDict()
        for w in all_workloads:
            model = CDAE().to(DEVICE)
            path = os.path.join(MODEL_DIR, f"{w}_gru_phase1.pth")
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.cdae_models[w] = model

        self.aggregator = InterferenceAggregator()

    def first_pass(self, base_inputs, wid, hidden):
        outputs = {}

        if hidden is None:
            hidden = {}

        for i, w in enumerate(wid):
            batch_size = base_inputs[i].shape[0]
            if w not in hidden or hidden[w] is None:
                hidden[w] = self._zero_hidden(batch_size)
            x = torch.cat([base_inputs[i], torch.zeros(batch_size, 1, 1, device=DEVICE)], dim=2)  # interference=0
            trace, _, _ = self.cdae_models[w](x, hidden[w])
            outputs[w] = trace.squeeze(1).detach()  # detach graph

        return outputs

    def second_pass(self, base_inputs, wid, hidden, first_outputs):
        outputs = {}
        new_hidden = {}

        interference = self.aggregator([first_outputs[w] for w in wid])  # global interference
        interference = interference.unsqueeze(1)  # shape [1,1,1]

        if hidden is None:
            hidden = {}

        for i, w in enumerate(wid):
            batch_size = base_inputs[i].shape[0]
            if w not in hidden or hidden[w] is None:
                hidden[w] = self._zero_hidden(batch_size)
            x = torch.cat([base_inputs[i], interference], dim=2)
            trace, _, h_new = self.cdae_models[w](x, hidden[w])
            outputs[w] = trace.squeeze(1)
            new_hidden[w] = h_new

        return outputs, new_hidden

    def _zero_hidden(self, batch_size):
        return torch.zeros(2, batch_size, HIDDEN, device=DEVICE)

# =========================================================
# Training
# =========================================================
def train():
    datasets = parse_trace_file(TRACE_FILE)
    model = Phase2Model().to(DEVICE)
    optimizer = optim.Adam([
        {"params": model.aggregator.parameters(), "lr": LR_AGG},
        {"params": model.cdae_models.parameters(), "lr": LR_CDAE}
    ])
    mse = nn.MSELoss()
    best_loss = float("inf")
    train_losses = []

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint['best_loss']
        print(f"Loaded best existing model and best_loss: {best_loss:.4f}.")

    for epoch in range(EPOCHS):
        total_epoch_loss = 0.0

        for data in datasets:
            wid = list(data.keys())
            T = len(next(iter(data.values()))["cpu"])
            inputs, targets = build_inputs_and_targets(data, wid)
            max_start = max(0, T - ROLLOUT_STEPS)
            num_segments = TRAIN_SEGMENTS_PER_DATASET or len(wid)
            if max_start > 0:
                start_vec = [0] + [
                    random.randint(1, max_start)
                    for _ in range(max(0, num_segments - 1))
                ]
            else:
                start_vec = [0]

            start_indices = torch.tensor(start_vec, dtype=torch.long, device=DEVICE)
            batch_size = start_indices.numel()

            optimizer.zero_grad()
            current_data_block_loss = torch.zeros((), device=DEVICE)
            hidden = {w: model._zero_hidden(batch_size) for w in wid}

            # Warm up each sampled segment to its own start index in one batched pass.
            for t in range(int(start_indices.max().item())):
                active_segments = (start_indices > t).nonzero(as_tuple=True)[0]
                if active_segments.numel() == 0:
                    continue

                active_hidden = {w: hidden[w][:, active_segments, :].contiguous() for w in wid}
                base_inputs = [
                    inputs[w][torch.full((active_segments.numel(),), t, dtype=torch.long, device=DEVICE)].unsqueeze(1)
                    for w in wid
                ]

                with torch.no_grad():
                    first_out = model.first_pass(base_inputs, wid, active_hidden)
                    _, updated_hidden = model.second_pass(base_inputs, wid, active_hidden, first_out)

                for w in wid:
                    hidden[w][:, active_segments, :] = updated_hidden[w].detach()

            # Roll out all sampled starts as one batch per workload-specific CDAE.
            for k in range(min(ROLLOUT_STEPS, T)):
                t_indices = start_indices + k
                valid_segments = (t_indices < T).nonzero(as_tuple=True)[0]
                if valid_segments.numel() == 0:
                    break

                step_indices = t_indices[valid_segments]
                active_hidden = {w: hidden[w][:, valid_segments, :].contiguous() for w in wid}
                base_inputs = [inputs[w][step_indices].unsqueeze(1) for w in wid]

                first_out = model.first_pass(base_inputs, wid, active_hidden)
                second_out, updated_hidden = model.second_pass(base_inputs, wid, active_hidden, first_out)

                for w in wid:
                    current_data_block_loss = current_data_block_loss + mse(
                        second_out[w],
                        targets[w][step_indices]
                    )
                    hidden[w] = updated_hidden[w]

            if current_data_block_loss.requires_grad:
                current_data_block_loss.backward()
                optimizer.step()

            total_epoch_loss += current_data_block_loss.item()

        train_losses.append(total_epoch_loss)
        print(f"Epoch {epoch} Loss {total_epoch_loss:.4f}")

        if total_epoch_loss < best_loss:
            best_loss = total_epoch_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
            }, SAVE_PATH)
            print("Best model and best_loss saved.")

    print("Training finished.")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase-2 Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR + "/training_loss.png")
    plt.close()

# =========================================================
# Evaluating
# =========================================================
def eval():
    datasets = parse_trace_file(TRACE_FILE)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    model = Phase2Model().to(DEVICE)
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    start = 20

    for data in datasets:
        wid = list(data.keys())
        T = len(data[wid[0]]["cpu"])
        hidden = None

        preds = {w: {"cpu": [], "cache": [], "mem": []} for w in wid}
        gts   = {w: {"cpu": [], "cache": [], "mem": []} for w in wid}
        last_pred = {w: None for w in wid}

        with torch.no_grad():
            for t in range(T):

                base_inputs = []

                for w in wid:
                    if t < start:
                        # teacher forcing
                        inp = build_input(data, w, t)
                    else:
                        # after start → autoregressive
                        if last_pred[w] is None:
                            inp = build_input(data, w, t)  # fallback on first step
                        else:
                            # Replace trace portion with predicted values
                            inp = build_input(data, w, t).clone()
                            inp[0:3] = last_pred[w]   # cpu, mem, cache prediction feeds back
                    base_inputs.append(inp)

                # First pass
                first_outputs = model.first_pass(base_inputs, wid, hidden)

                # Second pass
                second_outputs, hidden = model.second_pass(
                    base_inputs, wid, hidden, first_outputs
                )

                for w in wid:
                    pred = second_outputs[w].cpu().numpy()
                    gt_cpu   = data[w]["cpu"][t]
                    gt_cache = data[w]["cache"][t]
                    gt_mem   = data[w]["mem"][t]

                    preds[w]["cpu"].append(pred[0][0])
                    preds[w]["cache"].append(pred[0][1])
                    preds[w]["mem"].append(pred[0][2])

                    gts[w]["cpu"].append(gt_cpu)
                    gts[w]["cache"].append(gt_cache)
                    gts[w]["mem"].append(gt_mem)

        # -------- Plot --------
        n_w = len(wid)
        fig = plt.figure(figsize=(16, 4 * n_w))

        # CPU subplot
        for i, w in enumerate(wid):
            row = i * 3

            # -------- CPU --------
            ax = fig.add_subplot(n_w * 3, 1, row + 1)
            ax.plot(gts[w]["cpu"], label="GT CPU")
            ax.plot(preds[w]["cpu"], '--', label="Pred CPU")
            ax.set_title(f"{w} - CPU")
            ax.legend()
            ax.grid()

            # -------- MEM --------
            ax = fig.add_subplot(n_w * 3, 1, row + 2)
            ax.plot(gts[w]["mem"], label="GT MEM")
            ax.plot(preds[w]["mem"], '--', label="Pred MEM")
            ax.set_title(f"{w} - MEM")
            ax.legend()
            ax.grid()

            # -------- CACHE --------
            ax = fig.add_subplot(n_w * 3, 1, row + 3)
            ax.plot(gts[w]["cache"], label="GT CACHE")
            ax.plot(preds[w]["cache"], '--', label="Pred CACHE")
            ax.set_title(f"{w} - CACHE")
            ax.legend()
            ax.grid()

        plt.tight_layout()

        save_prefix = '_'.join(wid)
        out_path = IMAGES_DIR + f"/{save_prefix}.png"
        plt.savefig(out_path)
        plt.close()

        print(f"Saved evaluation plot → {out_path}")

# =========================================================
if __name__ == "__main__":
    if MODE == "train":
        train()
    eval()
