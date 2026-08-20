from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metermind.config import INTERVALS_PER_DAY, PATHS, RANDOM_SEED, SLOTS_COLUMNS

logger = logging.getLogger(__name__)

LATENT_DIM = 8
DEFAULT_N_PERSONAS = 6
MIN_DAILY_KWH = 0.5


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the shape autoencoder.\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc


def normalise_profiles(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    work = frame.copy()
    usable = pd.Series(True, index=work.index)
    for flag, sense in (
        ("is_modelling_day", True),
        ("flag_dst_day", False),
    ):
        if flag in work.columns:
            usable &= work[flag].astype(bool) if sense else ~work[flag].astype(bool)

    totals = work[SLOTS_COLUMNS].to_numpy(dtype=np.float64).sum(axis=1)
    usable &= totals > MIN_DAILY_KWH

    work = work.loc[usable].reset_index(drop=True)
    matrix = work[SLOTS_COLUMNS].to_numpy(dtype=np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    shapes = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    finite = np.isfinite(shapes).all(axis=1)
    logger.info(
        "Shape matrix: %s usable household-days from %s (dropped %s)",
        f"{int(finite.sum()):,}", f"{len(frame):,}", f"{len(frame) - int(finite.sum()):,}",
    )
    return shapes[finite].astype(np.float32), work.loc[finite].reset_index(drop=True)


def build_autoencoder(latent_dim: int = LATENT_DIM):
    torch = _torch()
    nn = torch.nn

    class CircularConv(nn.Module):

        def __init__(self, cin, cout, kernel=5, stride=1):
            super().__init__()
            self.pad = kernel // 2
            self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, padding=0)

        def forward(self, x):
            return self.conv(torch.nn.functional.pad(x, (self.pad, self.pad), mode="circular"))

    class ShapeAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                CircularConv(1, 16, 5), nn.GELU(),
                CircularConv(16, 32, 5, stride=2), nn.GELU(),
                CircularConv(32, 32, 5, stride=2), nn.GELU(),
                nn.Flatten(),
                nn.Linear(32 * 12, 64), nn.GELU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.GELU(),
                nn.Linear(64, 32 * 12), nn.GELU(),
                nn.Unflatten(1, (32, 12)),
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                CircularConv(32, 32, 5), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                CircularConv(32, 16, 5), nn.GELU(),
                CircularConv(16, 1, 5),
            )

        def forward(self, x):
            code = self.encoder(x)
            return self.decoder(code), code

        def encode(self, x):
            return self.encoder(x)

    return ShapeAutoencoder()


@dataclass
class TrainedShapeModel:
    state_dict: dict
    latent_dim: int
    epochs_run: int
    best_loss: float
    n_train: int

    def module(self):
        model = build_autoencoder(self.latent_dim)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def save(self, directory: Path | None = None) -> Path:
        torch = _torch()
        target = Path(directory) if directory else PATHS.artifacts / "shape_autoencoder"
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, target / "weights.pt")
        (target / "meta.json").write_text(json.dumps({
            "latent_dim": self.latent_dim,
            "epochs_run": self.epochs_run,
            "best_loss": self.best_loss,
            "n_train": self.n_train,
            "intervals_per_day": INTERVALS_PER_DAY,
            "normalisation": "profile divided by its own daily total (shape only)",
            "padding": "circular",
        }, indent=2))
        logger.info("Shape autoencoder written to %s", target)
        return target

    @classmethod
    def load(cls, directory: Path | None = None) -> TrainedShapeModel:
        torch = _torch()
        target = Path(directory) if directory else PATHS.artifacts / "shape_autoencoder"
        meta = json.loads((target / "meta.json").read_text())
        return cls(
            state_dict=torch.load(target / "weights.pt", map_location="cpu", weights_only=True),
            latent_dim=meta["latent_dim"],
            epochs_run=meta["epochs_run"],
            best_loss=meta["best_loss"],
            n_train=meta["n_train"],
        )


def train_autoencoder(
    shapes: np.ndarray,
    latent_dim: int = LATENT_DIM,
    epochs: int = 30,
    batch_size: int = 512,
    max_train: int = 250_000,
    quick: bool = False,
) -> TrainedShapeModel:
    torch = _torch()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(2)

    if quick:
        epochs, max_train = 5, 30_000

    rng = np.random.default_rng(RANDOM_SEED)
    if len(shapes) > max_train:
        idx = rng.choice(len(shapes), size=max_train, replace=False)
        train_x = shapes[idx]
        logger.info("  subsampled %s of %s days for fitting", f"{max_train:,}", f"{len(shapes):,}")
    else:
        train_x = shapes

    split = int(len(train_x) * 0.9)
    perm = rng.permutation(len(train_x))
    tr = torch.from_numpy(train_x[perm[:split]]).unsqueeze(1)
    va = torch.from_numpy(train_x[perm[split:]]).unsqueeze(1)

    model = build_autoencoder(latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  autoencoder parameters: %s", f"{n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = torch.nn.MSELoss()

    best_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tr))
        running = 0.0
        for start in range(0, len(tr), batch_size):
            batch = tr[order[start : start + batch_size]]
            optimiser.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * len(batch)
        train_loss = running / max(len(tr), 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(va)[0], va).item())
        scheduler.step()

        flag = ""
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        if epoch % 5 == 0 or epoch == 1 or flag:
            logger.info("    epoch %2d/%d  train %.6f  valid %.6f%s",
                        epoch, epochs, train_loss, val_loss, flag)

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedShapeModel(
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        latent_dim=latent_dim,
        epochs_run=epochs,
        best_loss=best_loss,
        n_train=int(len(tr)),
    )


def encode(model: TrainedShapeModel, shapes: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    torch = _torch()
    module = model.module()
    out = []
    with torch.no_grad():
        for start in range(0, len(shapes), batch_size):
            batch = torch.from_numpy(shapes[start : start + batch_size]).unsqueeze(1)
            out.append(module.encode(batch).numpy())
    return np.vstack(out) if out else np.empty((0, model.latent_dim), dtype=np.float32)


def reconstruction_error(
    model: TrainedShapeModel, shapes: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    torch = _torch()
    module = model.module()
    errors = []
    with torch.no_grad():
        for start in range(0, len(shapes), batch_size):
            batch = torch.from_numpy(shapes[start : start + batch_size]).unsqueeze(1)
            recon, _ = module(batch)
            errors.append(((recon - batch) ** 2).mean(dim=(1, 2)).numpy())
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


def choose_n_personas(codes: np.ndarray, candidates=(4, 5, 6, 7, 8), sample: int = 20_000) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(codes), size=min(sample, len(codes)), replace=False)
    subset = codes[idx]

    scores = {}
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=4, random_state=RANDOM_SEED).fit_predict(subset)
        scores[k] = round(float(silhouette_score(subset, labels)), 4)
        logger.info("    k=%d silhouette %.4f", k, scores[k])
    best = max(scores, key=scores.get)
    return {"scores": scores, "best_k": int(best)}


def describe_persona(mean_shape: np.ndarray, population: np.ndarray | None = None) -> str:
    bands = {
        "overnight": slice(0, 12),
        "early morning": slice(12, 18),
        "daytime": slice(18, 32),
        "evening peak": slice(32, 40),
        "late evening": slice(40, 48),
    }
    share = {name: float(mean_shape[window].sum()) for name, window in bands.items()}

    if population is None:
        dominant = max(share, key=share.get)
        return f"{dominant.capitalize()} dominant"

    baseline = {name: float(population[window].sum()) for name, window in bands.items()}
    lift = {name: share[name] - baseline[name] for name in share}
    strongest = max(lift, key=lift.get)
    weakest = min(lift, key=lift.get)

    peak_slot = int(np.argmax(mean_shape))
    peak_label = f"{peak_slot // 2:02d}:{'30' if peak_slot % 2 else '00'}"
    spread = float(np.std(mean_shape) * INTERVALS_PER_DAY)
    baseline_spread = float(np.std(population) * INTERVALS_PER_DAY)

    if spread < baseline_spread * 0.75:
        return f"Flat profile, little daily variation, peak {peak_label}"
    if lift[strongest] < 0.02:
        return f"Average profile, peak {peak_label}"

    names = {
        "overnight": "Overnight heavy",
        "early morning": "Early riser",
        "daytime": "Daytime occupancy",
        "evening peak": "Evening peaking",
        "late evening": "Late evening",
    }
    return f"{names[strongest]}, light on {weakest} (peak {peak_label})"


def assign_personas(codes: np.ndarray, n_personas: int = DEFAULT_N_PERSONAS, shapes=None):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_personas, n_init=10, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(codes)

    population = (
        shapes.mean(axis=0) if shapes is not None else np.full(INTERVALS_PER_DAY, 1 / INTERVALS_PER_DAY)
    )

    summary = []
    for cluster in range(n_personas):
        mask = labels == cluster
        mean_shape = shapes[mask].mean(axis=0) if shapes is not None else np.zeros(INTERVALS_PER_DAY)
        summary.append({
            "persona_id": cluster,
            "persona": describe_persona(mean_shape, population),
            "n_days": int(mask.sum()),
            "share_pct": round(100 * float(mask.mean()), 2),
            "peak_slot": int(np.argmax(mean_shape)),
            "peak_hour_local": round(float(np.argmax(mean_shape)) / 2, 1),
            "night_share": round(float(mean_shape[0:14].sum()), 4),
            "evening_share": round(float(mean_shape[32:40].sum()), 4),
            "mean_shape": [round(float(v), 6) for v in mean_shape],
        })

    return labels, kmeans, pd.DataFrame(summary)
