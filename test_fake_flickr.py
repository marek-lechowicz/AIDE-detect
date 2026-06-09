"""Evaluate pretrained AIDE models on the FakeFlickr dataset (Flickr30k test split).

This mirrors the protocol used by the sibling detectors
(``DIRE/test_fake_flickr.py`` and ``UniversalFakeDetect/evaluate_on_fake_flickr.py``)
so the numbers are directly comparable. For every generator under
``<dataset_root>/generated/<gen>/img`` this script:

  1. Filters images to the IDs listed in the Flickr30k test split.
  2. Pairs them with the matching real images from ``<dataset_root>/real``
     (or ``real_rescaled`` for ``flux_fill_real_rescaled``, which was
     conditioned on the rescaled reals).
  3. Runs them through AIDE's exact test-time preprocessing (DCT band
     selection -> 256x256 resize + ImageNet normalization, producing the
     5-view tensor the model expects).
  4. Loads a pretrained AIDE checkpoint (``--ckpt``) and reports
     ACC / AP / R_ACC / F_ACC over class-1 (fake) softmax probabilities.

To remove the real-JPEG vs fake-PNG/WebP format confound, non-JPEG inputs
are re-encoded to JPEG quality 90 before preprocessing (the FakeFlickr eval
protocol). Disable with ``--no-jpeg-equalize``.

Results are written as one CSV row per generator.

Example
-------
    .venv/bin/python test_fake_flickr.py \
        --ckpt models_files/progan_train-002.pth \
        --results-csv data/results/fake_flickr_aide_progan.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data.dct import DCT_base_Rec_Module  # noqa: E402
from models.AIDE import AIDE  # noqa: E402

DEFAULT_GENERATORS = [
    "sd_1_5",
    "sd_3_5_large",
    "sdxl_turbo",
    "z_image_turbo",
    "flux_1_dev",
    "flux_fill_flux_1_dev",
    "flux_fill_sd_3_5_large",
    "flux_fill_real_rescaled",
]

# Generators conditioned on the rescaled-real source images. For these the
# matching "real" is the rescaled PNG, not the original JPG.
RESCALED_REAL_GENS = {"flux_fill_real_rescaled"}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".PNG", ".JPG")

# AIDE test-time transforms, copied verbatim from data/datasets.py::TestDataset
# so preprocessing matches training exactly.
_to_tensor = transforms.ToTensor()
_resize_norm = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_test_ids(split_file: Path) -> list[str]:
    with split_file.open("r") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise RuntimeError(f"Test split is empty: {split_file}")
    return ids


def find_image(dirpath: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        cand = dirpath / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def load_image(path: Path, jpeg_equalize: bool) -> Image.Image:
    """Open an image as RGB, optionally re-encoding non-JPEG inputs to JPEG q90.

    Mirrors the FakeFlickr eval protocol: original JPEG reals pass through
    unchanged, lossless fakes / rescaled reals get the same q=90 compression
    so the classifier cannot cheat on the format/compression artifact.
    """
    if jpeg_equalize and path.suffix.lower() not in (".jpg", ".jpeg"):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read {path}")
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()
    return img.convert("RGB")


class FlickrDataset(Dataset):
    """Yields AIDE's 5-view tensor stack for a list of (path, label) pairs."""

    def __init__(self, items: list[tuple[Path, int]], jpeg_equalize: bool):
        self.items = items
        self.jpeg_equalize = jpeg_equalize
        self.dct = DCT_base_Rec_Module()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = load_image(path, self.jpeg_equalize)
        image = _to_tensor(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)

        x_0 = _resize_norm(image)
        x_minmin = _resize_norm(x_minmin)
        x_maxmax = _resize_norm(x_maxmax)
        x_minmin1 = _resize_norm(x_minmin1)
        x_maxmax1 = _resize_norm(x_maxmax1)

        stack = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)
        return stack, torch.tensor(int(label))


def build_items(
    dataset_root: Path,
    gen: str,
    test_ids: list[str],
) -> list[tuple[Path, int]]:
    """Collect (path, label) pairs: reals (label 0) + this generator's fakes (label 1)."""
    real_name = "real_rescaled" if gen in RESCALED_REAL_GENS else "real"
    real_dir = dataset_root / real_name
    fake_dir = dataset_root / "generated" / gen / "img"
    if not real_dir.is_dir():
        raise FileNotFoundError(f"Real folder not found: {real_dir}")
    if not fake_dir.is_dir():
        raise FileNotFoundError(f"Fake folder not found: {fake_dir}")

    items: list[tuple[Path, int]] = []
    n_real = n_fake = 0
    for img_id in test_ids:
        real = find_image(real_dir, img_id)
        fake = find_image(fake_dir, img_id)
        # Only keep paired samples so real/fake counts stay balanced per ID.
        if real is None or fake is None:
            continue
        items.append((real, 0))
        items.append((fake, 1))
        n_real += 1
        n_fake += 1
    if not items:
        raise RuntimeError(f"No paired test-split images found for generator {gen}")
    print(f"  matched {n_real} real / {n_fake} fake ({real_name})")
    return items


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: FlickrDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, target in tqdm(loader, desc="  infer", dynamic_ncols=True, leave=False):
        images = images.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(images)
        else:
            output = model(images)
        # class-1 (fake) probability
        p = torch.softmax(output.float(), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = y_prob > 0.5
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "AP": float(average_precision_score(y_true, y_prob)),
        "R_ACC": float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])),
        "F_ACC": float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])),
        "N_real": int((y_true == 0).sum()),
        "N_fake": int((y_true == 1).sum()),
    }


def load_model(ckpt: Path, device: torch.device) -> torch.nn.Module:
    """Build AIDE and load a pretrained checkpoint.

    The architecture is built with ``resnet_path=None``/``convnext_path=None``
    (no ImageNet/CLIP pre-init needed) because every weight -- including the
    frozen ConvNeXt-XXL backbone -- is restored from the AIDE checkpoint.
    """
    print("Building AIDE model ...")
    model = AIDE(resnet_path=None, convnext_path=None)
    checkpoint = torch.load(ckpt, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    # Strip a possible DDP 'module.' prefix.
    state = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model.to(device).eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Pretrained AIDE checkpoint (.pth), e.g. models_files/progan_train-002.pth.")
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/fake-flickr"),
                   help="Root of the fake-flickr dataset.")
    p.add_argument("--test-split", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/flickr30k_entities/test.txt"),
                   help="File with one Flickr30k image ID per line (the test split).")
    p.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS,
                   help=f"Generator subdirs to evaluate (default: {DEFAULT_GENERATORS}).")
    p.add_argument("--results-csv", type=Path,
                   default=REPO_ROOT / "data" / "results" / "fake_flickr_aide.csv",
                   help="Output CSV path.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-jpeg-equalize", dest="jpeg_equalize", action="store_false", default=True,
                   help="Disable JPEG-q90 equalization of non-JPEG inputs (default: ON, "
                        "to remove the real-JPEG vs fake-PNG/WebP format confound).")
    p.add_argument("--no-amp", dest="use_amp", action="store_false", default=True,
                   help="Disable bf16 autocast inference (default: ON).")
    p.add_argument("--debug", action="store_true",
                   help="Only run on --debug-samples IDs per generator to smoke-test.")
    p.add_argument("--debug-samples", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if not args.test_split.is_file():
        raise FileNotFoundError(f"--test-split not found: {args.test_split}")

    test_ids = read_test_ids(args.test_split)
    print(f"Loaded {len(test_ids)} test IDs from {args.test_split}")
    if args.debug:
        test_ids = test_ids[: args.debug_samples]
        print(f"[DEBUG] truncated to {len(test_ids)} IDs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.ckpt, device)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for gen in args.generators:
        print(f"\n=== {gen} ===")
        items = build_items(args.dataset_root, gen, test_ids)
        dataset = FlickrDataset(items, jpeg_equalize=args.jpeg_equalize)
        y_prob, y_true = run_inference(
            model, dataset, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.use_amp,
        )
        metrics = compute_metrics(y_true, y_prob)
        print("  " + gen + ": " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()))
        rows.append({"generator": gen, **metrics})

    fieldnames = ["generator", "ACC", "AP", "R_ACC", "F_ACC", "N_real", "N_fake"]
    with args.results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {args.results_csv}")


if __name__ == "__main__":
    main()
