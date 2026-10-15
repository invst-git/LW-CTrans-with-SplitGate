#!/usr/bin/env python3
"""Convert AISD into the nnU-Net v2 raw layout used by LW-CTrans.

Why this is not a one-liner
---------------------------
AISD ships two things that must be combined:

  * image.zip / mask.zip -- per-patient folders of per-SLICE 2D PNGs.
    The mask PNGs hold the real integer labels {0,1,2,3,4,5}. The image
    PNGs are 8-bit windowed exports: the true Hounsfield Units have been
    destroyed, and PNG carries no voxel spacing. Unusable as CT input.

  * dicom-*.tar.gz -- per-patient <id>/CT and <id>/DWI DICOM folders.
    These carry real HU (RescaleSlope/Intercept) and real per-patient
    spacing. Spacing genuinely varies per patient (in-plane ~0.40-0.44mm,
    slice thickness 3-9mm) -- it is NOT the flat 5mm the README implies.

So: CT volume comes from DICOM, labels come from the PNG masks, and the
two are matched per patient. Slice ORDER between the sources is verified
by cross-correlating DICOM pixel data against the PNG *image* slices in
both possible orderings, before anything is written.

Usage -- always dry run first:

    python convert_aisd.py --dry_run \
        --dicom_roots aisd_raw_dicom/dicom-0 aisd_raw_dicom/dicom-1 aisd_raw_dicom/dicom-2 \
        --image_root aisd_raw/image --mask_root aisd_raw/mask \
        --raw_root "$nnUNet_raw" --log alignment_dry.csv

Check the summary, then drop --dry_run to write.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image

# The 52 test-set patient IDs published in the AISD README.
TEST_IDS = {
    "0073410", "0072723", "0226290", "0537908", "0538058", "0091415", "0538780",
    "0073540", "0226188", "0226258", "0226314", "0091507", "0226298", "0538975",
    "0226257", "0226142", "0072681", "0091538", "0538983", "0537961", "0091646",
    "0072765", "0226137", "0091621", "0091458", "0021822", "0538319", "0226133",
    "0091657", "0537925", "0073489", "0538502", "0091476", "0226136", "0538532",
    "0073312", "0539025", "0226309", "0226307", "0091383", "0021092", "0537990",
    "0226299", "0073060", "0538505", "0073424", "0091534", "0226125", "0072691",
    "0538425", "0226199", "0226261",
}


def load_ct_series(folder: Path):
    slices = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except Exception:
            continue
        if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelData"):
            slices.append(ds)
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    return slices


def hu_volume(slices) -> np.ndarray:
    out = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        out.append(arr * slope + intercept)
    return np.stack(out, axis=0)


def z_spacing(slices) -> float:
    zs = [float(d.ImagePositionPatient[2]) for d in slices]
    if len(zs) < 2:
        return float(getattr(slices[0], "SliceThickness", 1.0))
    return float(np.median(np.abs(np.diff(zs))))


def brain_window(hu: np.ndarray, center=40.0, width=100.0) -> np.ndarray:
    lo, hi = center - width / 2, center + width / 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = np.asarray(
            Image.fromarray((b * 255).astype(np.uint8)).resize((a.shape[1], a.shape[0]))
        ).astype(np.float32) / 255.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.sum(a * b) / denom)


def png_stack(folder: Path):
    files = list(folder.glob("*.png"))
    try:
        return sorted(files, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(files)


def process(pid, dicom_roots, image_root, mask_root, ds_folder, keep, dry_run, name_prefix):
    ct_dir = None
    for root in dicom_roots:
        cand = root / pid / "CT"
        if cand.exists():
            ct_dir = cand
            break
    if ct_dir is None:
        return {"pid": pid, "status": "no_dicom"}

    slices = load_ct_series(ct_dir)
    img_png = png_stack(image_root / pid)
    msk_png = png_stack(mask_root / pid)
    n = len(slices)

    if n == 0 or len(img_png) != n or len(msk_png) != n:
        return {"pid": pid, "status": "count_mismatch", "n_dicom": n,
                "n_image_png": len(img_png), "n_mask_png": len(msk_png)}

    volume = hu_volume(slices)
    first_bw, last_bw = brain_window(volume[0]), brain_window(volume[-1])
    p_first = np.asarray(Image.open(img_png[0])).astype(np.float32) / 255.0
    p_last = np.asarray(Image.open(img_png[-1])).astype(np.float32) / 255.0

    fwd = ncc(first_bw, p_first) + ncc(last_bw, p_last)
    rev = ncc(first_bw, p_last) + ncc(last_bw, p_first)
    forward = fwd >= rev
    gap = abs(fwd - rev) / max(abs(fwd), abs(rev), 1e-6)
    flagged = gap < 0.15

    row = {"pid": pid, "status": "flagged" if flagged else "ok", "n_dicom": n,
           "orientation": "forward" if forward else "reverse",
           "score_forward": round(fwd, 4), "score_reverse": round(rev, 4),
           "confidence_gap": round(gap, 4)}

    if dry_run or flagged:
        return row

    order = list(range(n)) if forward else list(range(n))[::-1]
    sx, sy = [float(v) for v in slices[0].PixelSpacing]
    sz = z_spacing(slices)

    img = sitk.GetImageFromArray(volume.astype(np.int16))
    img.SetSpacing((sx, sy, sz))
    img.SetOrigin(tuple(float(v) for v in slices[0].ImagePositionPatient))

    split = "Ts" if pid in TEST_IDS else "Tr"
    stem = f"{name_prefix}_{pid}"
    sitk.WriteImage(img, str(ds_folder / f"images{split}" / f"{stem}_0000.nii.gz"), True)

    lab = np.stack([np.array(Image.open(msk_png[i])) for i in order], axis=0)
    lab = np.isin(lab, list(keep)).astype(np.uint8)
    lab_img = sitk.GetImageFromArray(lab)
    lab_img.CopyInformation(img)
    sitk.WriteImage(lab_img, str(ds_folder / f"labels{split}" / f"{stem}.nii.gz"), True)

    row["split"] = split
    row["lesion_voxels"] = int(lab.sum())
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_roots", nargs="+", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--dataset_id", type=int, default=402)
    ap.add_argument("--dataset_name", default="AISD")
    ap.add_argument("--acute_only", action="store_true",
                    help="lesion = labels {2,3,5}. Default = {1,2,3,5} per the AISD README.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--log", default="alignment_log.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Guard against the empty-variable trap: an unset $nnUNet_raw expands to
    # "" and would silently write into the current directory.
    if not args.raw_root.strip():
        sys.exit("ERROR: --raw_root is empty. Did you forget to export nnUNet_raw?\n"
                 "       Run: source ~/lwctrans_env.sh")
    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        sys.exit(f"ERROR: --raw_root does not exist: {raw_root}")

    keep = {2, 3, 5} if args.acute_only else {1, 2, 3, 5}
    print(f"lesion label set: {sorted(keep)}"
          f"{'  (--acute_only)' if args.acute_only else '  (AISD README default)'}")

    dicom_roots = [Path(r).expanduser() for r in args.dicom_roots]
    for r in dicom_roots:
        if not r.is_dir():
            sys.exit(f"ERROR: dicom root does not exist: {r}")
    image_root, mask_root = Path(args.image_root), Path(args.mask_root)

    ds_folder = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if not args.dry_run:
        for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (ds_folder / sub).mkdir(parents=True, exist_ok=True)

    patients = sorted(p.name for p in image_root.iterdir() if p.is_dir())
    if args.limit:
        patients = patients[: args.limit]
    print(f"{len(patients)} patients under {image_root}")

    rows, n_train = [], 0
    for i, pid in enumerate(patients, 1):
        r = process(pid, dicom_roots, image_root, mask_root, ds_folder,
                    keep, args.dry_run, args.dataset_name)
        rows.append(r)
        if r.get("split") == "Tr":
            n_train += 1
        if i % 50 == 0:
            print(f"  {i}/{len(patients)}")

    fields = ["pid", "status", "n_dicom", "n_image_png", "n_mask_png", "orientation",
              "score_forward", "score_reverse", "confidence_gap", "split", "lesion_voxels"]
    with open(args.log, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    status = {}
    orient = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
        if "orientation" in r:
            orient[r["orientation"]] = orient.get(r["orientation"], 0) + 1
    print(f"\nstatus:      {status}")
    print(f"orientation: {orient}")
    print(f"log:         {args.log}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        print("Expected: status all 'ok' apart from a few no_dicom/count_mismatch,")
        print("and orientation 100% 'forward'. Any 'flagged' rows need a look before you write.")
        return

    empty = sum(1 for r in rows if r.get("lesion_voxels") == 0)
    with open(ds_folder / "dataset.json", "w") as fh:
        json.dump({"channel_names": {"0": "CT"},
                   "labels": {"background": 0, "lesion": 1},
                   "numTraining": n_train,
                   "file_ending": ".nii.gz"}, fh, indent=4)
    print(f"\nwrote {n_train} training cases to {ds_folder}")
    if empty:
        print(f"note: {empty} converted case(s) have zero lesion voxels after label mapping")


if __name__ == "__main__":
    main()
