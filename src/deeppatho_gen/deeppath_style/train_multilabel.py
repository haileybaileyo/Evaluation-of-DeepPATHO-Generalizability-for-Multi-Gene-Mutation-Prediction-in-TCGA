"""DeepPATH-style multi-output Inception v3 in PyTorch (Stage A: TCGA-LUAD DX slides).

A faithful re-implementation of the *method* of Coudray et al. 2018 (Nat Med),
following the authors' TensorFlow code (ncoudray/DeepPATH, 01_training/xClasses):

  model      Inception v3 trained FROM SCRATCH, 299x299 input scaled to [-1, 1],
             one sigmoid output per gene (10 genes at once), auxiliary head with
             loss weight 0.4, label smoothing 0.1, BN decay 0.9997 / eps 1e-3
  optimizer  TF-style RMSProp (decay 0.9, momentum 0.9, epsilon 1.0, ms slot
             initialised to 1), lr 0.1 decayed x0.16 every 30 epochs, L2 weight
             decay 4e-5 on conv/fc weights, EMA of weights (0.9999) used for eval
  training   random crop (area 10-100%, aspect 0.75-1.33) -> 299, horizontal
             flip, colour distortion (brightness +-32/255, saturation 0.5-1.5,
             hue +-0.2, contrast 0.5-1.5); eval: plain resize to 299
  testing    per-tile probabilities aggregated per slide by mean probability
             (and by the fraction of tiles with p >= 0.5); AUC per gene

Differences from the original (all deliberate and recorded in results.json):
  * PyTorch, TF32 matmuls; BN momentum 0.01 instead of 0.0003 (short runs; with 0.001
    eval-mode outputs were saturated for the first few hundred steps)
  * split: OUR patient-level stratified 70/15/15 (split_seed 42), i.e. the same
    71 test patients as every other experiment here (the original split is a
    random, unseeded shuffle that cannot be reproduced)
  * tiles: slide-capped at --max-tiles-per-slide at tiling time; no separate
    "LUAD tile" classifier in Stage A (all non-background tiles are used)
  * labels: cBioPortal PanCancer Atlas calls for the same 10 genes
  * run length: --max-steps (default 60k) with best-validation-loss checkpoint
    instead of 500k steps

Usage:
    python -m deeppatho_gen.deeppath_style.train_multilabel --config configs/deeppath_style/stageA_dx.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3

from ..metrics import binary_metrics, find_best_threshold
from ..slide_level.fixed_split_logreg import get_fixed_split

GENES = ["TP53", "LRP1B", "KRAS", "KEAP1", "FAT4", "STK11", "EGFR", "FAT1", "NF1", "SETBP1"]
EGFR_IDX = GENES.index("EGFR")


# ----------------------------------------------------------------- data
def distort_color(x: torch.Tensor) -> torch.Tensor:
    """TF inception `distort_color` (two op orders), x float CHW in [0, 1]."""
    bri = lambda t: t + random.uniform(-32 / 255, 32 / 255)  # noqa: E731
    sat = lambda t: TF.adjust_saturation(t.clamp(0, 1), random.uniform(0.5, 1.5))  # noqa: E731
    hue = lambda t: TF.adjust_hue(t.clamp(0, 1), random.uniform(-0.2, 0.2))  # noqa: E731
    con = lambda t: TF.adjust_contrast(t.clamp(0, 1), random.uniform(0.5, 1.5))  # noqa: E731
    for op in ([bri, sat, hue, con] if random.random() < 0.5 else [bri, con, sat, hue]):
        x = op(x)
    return x.clamp(0, 1)


class TileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, y: np.ndarray, train: bool, size: int = 299):
        self.paths = [str(root / p) for p in df["path"]]
        self.y = y.astype(np.float32)
        self.train, self.size = train, size
        self.crop = T.RandomResizedCrop(size, scale=(0.1, 1.0), ratio=(0.75, 4 / 3),
                                        interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.train:
            img = self.crop(img)
            if random.random() < 0.5:
                img = TF.hflip(img)
            x = distort_color(TF.to_tensor(img))
        else:
            x = TF.to_tensor(img.resize((self.size, self.size), Image.BILINEAR))
        return (x - 0.5) * 2.0, torch.from_numpy(self.y[i]), i


# ----------------------------------------------------------------- model/optim
def build_model(n_out: int, bn_momentum: float) -> nn.Module:
    m = inception_v3(weights=None, aux_logits=True, transform_input=False, init_weights=True, num_classes=n_out)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = bn_momentum  # eps is already 1e-3
    return m


class TFRMSprop:
    """tf.train.RMSPropOptimizer semantics: ms init 1, step = lr*g/sqrt(ms+eps)
    accumulated into a momentum buffer; coupled L2 on tensors with ndim > 1."""

    def __init__(self, params, decay=0.9, momentum=0.9, eps=1.0, weight_decay=4e-5):
        self.params = [p for p in params if p.requires_grad]
        self.decay, self.momentum, self.eps, self.wd = decay, momentum, eps, weight_decay
        self.ms = [torch.ones_like(p) for p in self.params]
        self.mom = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, lr: float):
        for p, ms, mom in zip(self.params, self.ms, self.mom):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p if (self.wd and p.ndim > 1) else p.grad
            ms.mul_(self.decay).addcmul_(g, g, value=1 - self.decay)
            mom.mul_(self.momentum).addcdiv_(g, (ms + self.eps).sqrt(), value=lr)
            p.sub_(mom)

    def state_dict(self):
        return {"ms": self.ms, "mom": self.mom}

    def load_state_dict(self, s):
        for a, b in zip(self.ms, s["ms"]):
            a.copy_(b)
        for a, b in zip(self.mom, s["mom"]):
            a.copy_(b)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.shadow, self.decay = copy.deepcopy(model).eval(), decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        d = min(self.decay, (1 + step) / (10 + step))  # TF num_updates warm-up
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            if s.dtype.is_floating_point:  # BN running mean/var are averaged too (as TF's moving-average variables)
                s.mul_(d).add_(b, alpha=1 - d)
            else:
                s.copy_(b)


def loss_fn(out, y, smoothing: float, aux_weight: float):
    ys = y * (1 - smoothing) + smoothing / y.shape[1]
    return (F.binary_cross_entropy_with_logits(out.logits, ys)
            + aux_weight * F.binary_cross_entropy_with_logits(out.aux_logits, ys))


# ----------------------------------------------------------------- evaluation
def safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def predict_tiles(model, df, root, labels, device, size, workers, batch=128):
    ds = TileDataset(df, root, labels, train=False, size=size)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval()
    out = []
    for x, _, _ in dl:
        out.append(torch.sigmoid(model(x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last))).float().cpu())
    return torch.cat(out).numpy()


def slide_table(df, probs, labels):
    t = pd.DataFrame(probs, columns=GENES)
    t["file_id"], t["case_id"] = df["file_id"].values, df["case_id"].values
    L = pd.DataFrame(labels, columns=[f"y_{g}" for g in GENES])
    L["file_id"] = df["file_id"].values
    mean = t.groupby("file_id")[GENES].mean()
    frac = t.assign(**{g: (t[g] >= 0.5).astype(float) for g in GENES}).groupby("file_id")[GENES].mean()
    lab = L.groupby("file_id").first()
    case = t.groupby("file_id")["case_id"].first()
    return mean, frac, lab, case


def summarize(df, probs, labels, n_boot=1000, seed=0):
    mean, frac, lab, _ = slide_table(df, probs, labels)
    rng = np.random.default_rng(seed)
    res = {}
    for gi, g in enumerate(GENES):
        y_s = lab[f"y_{g}"].values
        ci = [float("nan")] * 2
        if len(np.unique(y_s)) == 2:
            a = []
            for _ in range(n_boot):
                i = rng.integers(0, len(y_s), len(y_s))
                if len(np.unique(y_s[i])) == 2:
                    a.append(roc_auc_score(y_s[i], mean[g].values[i]))
            if a:
                ci = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        res[g] = {"n_slides": int(len(y_s)), "n_pos_slides": int(y_s.sum()),
                  "slide_auc_mean_prob": safe_auc(y_s, mean[g].values), "slide_auc_ci95": ci,
                  "slide_auc_frac_tiles": safe_auc(y_s, frac[g].values),
                  "tile_auc": safe_auc(labels[:, gi], probs[:, gi])}
    return res


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check on whatever tiles exist")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg.update(max_steps=cfg.get("smoke_steps", 40), eval_every=cfg.get("smoke_eval_every", 20), batch_size=16, num_workers=4,
                   val_tiles_per_slide=20, patience_evals=0, ckpt_every=10)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out = Path(cfg["output_dir"] + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["tiles_dir"])

    # ---- data, labels, split
    tiles = pd.read_csv(root / cfg["tile_index"])
    lab10 = pd.read_csv(cfg["labels_10gene"]).groupby("case_id")[GENES].max()
    tiles = tiles[tiles.case_id.isin(lab10.index)].reset_index(drop=True)
    if args.smoke:
        order = {c: ["train", "train", "val", "test"][i % 4] for i, c in enumerate(sorted(tiles.case_id.unique()))}
        split = order
    else:
        split, egfr_labels = get_fixed_split(cfg["split_coords"], cfg["split_labels"], "EGFR",
                                             cfg["train_frac"], cfg["val_frac"], cfg["split_seed"])
        chk = {c: int(lab10.loc[c, "EGFR"]) for c in egfr_labels if c in lab10.index}
        if cfg.get("check_egfr_labels", True):  # set false only when EGFR is deliberately re-defined (see define_egfr_variants.py)
            assert all(chk[c] == egfr_labels[c] for c in chk), "EGFR labels differ between label files"
        tiles = tiles[tiles.case_id.isin(split)].reset_index(drop=True)
    tiles["split"] = tiles.case_id.map(split)
    Y = lab10.loc[tiles.case_id, GENES].values.astype(np.float32)
    parts = {k: np.where(tiles.split.values == k)[0] for k in ["train", "val", "test"]}
    for k, ix in parts.items():
        d = tiles.iloc[ix]
        print(f"{k:5s}: {d.case_id.nunique():3d} patients, {d.file_id.nunique():3d} slides, {len(d):7,} tiles | "
              f"EGFR+ patients {int(Y[ix, EGFR_IDX][pd.Series(d.case_id.values).drop_duplicates().index].sum())}", flush=True)
    rng = np.random.default_rng(cfg["seed"])
    val_ix = np.concatenate([g.index.values if len(g) <= cfg["val_tiles_per_slide"] else
                             rng.choice(g.index.values, cfg["val_tiles_per_slide"], replace=False)
                             for _, g in tiles.iloc[parts["val"]].groupby("file_id")])
    val_df, val_Y = tiles.loc[val_ix], Y[val_ix]
    tr = tiles.iloc[parts["train"]]
    train_ds = TileDataset(tr, root, Y[parts["train"]], train=True, size=cfg["image_size"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True, prefetch_factor=4)
    steps_per_epoch = len(train_ds) // cfg["batch_size"]
    decay_steps = int(cfg["epochs_per_decay"] * steps_per_epoch)
    print(f"{steps_per_epoch} steps/epoch; lr decay x{cfg['lr_decay']} every {decay_steps} steps; "
          f"max_steps {cfg['max_steps']} = {cfg['max_steps'] / steps_per_epoch:.1f} epochs", flush=True)

    # ---- model
    model = build_model(len(GENES), cfg["bn_momentum"]).to(device).to(memory_format=torch.channels_last)
    opt = TFRMSprop(model.parameters(), cfg["rms_decay"], cfg["momentum"], cfg["rms_eps"], cfg["weight_decay"])
    ema = EMA(model, cfg["ema_decay"])
    step, best, bad, history = 0, {"val_loss": float("inf"), "step": 0}, 0, []
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, best, bad, history = ck["step"], ck["best"], ck["bad"], ck["history"]
        print(f"resumed from step {step}", flush=True)
    (out / "run_config.json").write_text(json.dumps({**cfg, "smoke": args.smoke, "genes": GENES}, indent=2))

    def save_last():
        """Crash-safe: write to a temp file, then atomically replace last.pt."""
        torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best": best, "bad": bad, "history": history}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")

    def evaluate(m):
        p = predict_tiles(m, val_df, root, val_Y, device, cfg["image_size"], cfg["num_workers"])
        bce = float(F.binary_cross_entropy(torch.from_numpy(p).clamp(1e-6, 1 - 1e-6), torch.from_numpy(val_Y)))
        s = summarize(val_df, p, val_Y, n_boot=0 + 1)
        return bce, s

    t0, run_loss, n_loss = time.time(), 0.0, 0
    done = step >= cfg["max_steps"]
    while not done:
        for x, y, _ in loader:
            model.train()
            lr = cfg["lr"] * cfg["lr_decay"] ** (step // decay_steps)
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y, cfg["label_smoothing"], cfg["aux_weight"])
            opt.zero_grad()
            loss.backward()
            opt.step(lr)
            step += 1
            ema.update(model, step)
            run_loss += loss.item(); n_loss += 1
            if step % 200 == 0:
                print(f"step {step} loss {run_loss / n_loss:.4f} lr {lr:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if not np.isfinite(run_loss):
                    raise SystemExit("loss diverged (NaN/inf)")
                run_loss, n_loss = 0.0, 0
            if step % cfg["eval_every"] == 0 or step >= cfg["max_steps"]:
                vl, s = evaluate(ema.shadow)
                row = {"step": step, "val_bce": vl, "egfr_slide_auc": s["EGFR"]["slide_auc_mean_prob"],
                       "egfr_tile_auc": s["EGFR"]["tile_auc"],
                       "mean_slide_auc_10genes": float(np.nanmean([v["slide_auc_mean_prob"] for v in s.values()])),
                       "minutes": (time.time() - t0) / 60}
                history.append(row)
                print("EVAL", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
                if vl < best["val_loss"]:
                    best, bad = {"val_loss": vl, "step": step}, 0
                    torch.save(ema.shadow.state_dict(), out / "best_ema.pt")
                else:
                    bad += 1
                save_last()
                (out / "history.json").write_text(json.dumps(history, indent=2))
                if step >= cfg["max_steps"] or (cfg["patience_evals"] and bad >= cfg["patience_evals"]):
                    done = True
                    break
            elif step % cfg.get("ckpt_every", 2000) == 0:
                save_last()  # frequent resume point between evaluations
        if step >= cfg["max_steps"]:
            done = True

    # ---- final evaluation with the best (lowest val BCE) EMA checkpoint
    print(f"\nbest val BCE {best['val_loss']:.4f} at step {best['step']}; evaluating test ...", flush=True)
    model.load_state_dict(torch.load(out / "best_ema.pt", map_location=device))
    res = {"best": best, "config": cfg, "smoke": args.smoke, "genes": GENES}
    all_df = {k: tiles.iloc[ix] for k, ix in parts.items()}
    probs = {k: predict_tiles(model, all_df[k], root, Y[parts[k]], device, cfg["image_size"], cfg["num_workers"])
             for k in ["val", "test"]}
    res["val"] = summarize(all_df["val"], probs["val"], Y[parts["val"]])
    res["test"] = summarize(all_df["test"], probs["test"], Y[parts["test"]])
    # EGFR in the project's usual protocol: threshold fit on VAL slides (F1-max), applied to TEST slides
    mv, _, lv, _ = slide_table(all_df["val"], probs["val"], Y[parts["val"]])
    mt, ft, lt, ct = slide_table(all_df["test"], probs["test"], Y[parts["test"]])
    yv, yt = lv["y_EGFR"].values.astype(int), lt["y_EGFR"].values.astype(int)
    thr = find_best_threshold(yv, mv["EGFR"].values)
    n = len(yt)
    def pred_ratio(m):
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        return (tp + fp) / n
    res["EGFR_test_slide_protocol"] = {
        "threshold_fit_on_val": float(thr),
        "at_val_threshold": binary_metrics(yt, mt["EGFR"].values, thr),
        "at_0.5": binary_metrics(yt, mt["EGFR"].values, 0.5),
    }
    for k in ["at_val_threshold", "at_0.5"]:
        res["EGFR_test_slide_protocol"][k]["predicted_positive_ratio"] = pred_ratio(res["EGFR_test_slide_protocol"][k])
    mt.assign(case_id=ct, **{f"y_{g}": lt[f"y_{g}"] for g in GENES}).to_csv(out / "test_slide_mean_probs.csv")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\n=== TEST slide-level AUC per gene (mean probability; 95% bootstrap CI over slides) ===")
    for g in GENES:
        r = res["test"][g]
        print(f"{g:7s} slides {r['n_slides']} (+{r['n_pos_slides']:2d})  AUC {r['slide_auc_mean_prob']:.3f} "
              f"[{r['slide_auc_ci95'][0]:.3f}-{r['slide_auc_ci95'][1]:.3f}]  frac-tiles AUC {r['slide_auc_frac_tiles']:.3f}  tile AUC {r['tile_auc']:.3f}")
    e = res["EGFR_test_slide_protocol"]
    for k in ["at_val_threshold", "at_0.5"]:
        m = e[k]
        print(f"EGFR @{k} (thr {m['threshold']:.3f}): P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
              f"pred+ {100 * m['predicted_positive_ratio']:.0f}% cm {m['confusion_matrix']}")
    print(f"\nWrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
