#!/usr/bin/env bash
# Convert AISD -> nnU-Net raw, then preprocess.
# Run AFTER `source ~/lwctrans_env.sh`:   bash 03_convert.sh
set -euo pipefail

DATA="$HOME/lwctrans-work/data"
cd "$DATA"

: "${nnUNet_raw:?nnUNet_raw not set -- run: source ~/lwctrans_env.sh}"
: "${nnUNet_preprocessed:?nnUNet_preprocessed not set -- run: source ~/lwctrans_env.sh}"
echo ">>> nnUNet_raw          = $nnUNet_raw"
echo ">>> nnUNet_preprocessed = $nnUNet_preprocessed"

ROOTS=()
for d in aisd_raw_dicom/dicom-0 aisd_raw_dicom/dicom-1 aisd_raw_dicom/dicom-2; do
  [ -d "$d" ] && ROOTS+=("$d")
done
echo ">>> dicom roots: ${ROOTS[*]}"

# ---- LESION LABEL SET -------------------------------------------------
# Default {1,2,3,5} follows the AISD README's definition of infarct lesion.
# Set ACUTE_ONLY=1 for {2,3,5}, which drops label 1 (remote/old infarct).
# Both conventions appear in published work; LW-CTrans does not ship the
# dataset code, so this cannot be resolved from the repo. Pick one and keep
# it fixed across every comparison you run.
ACUTE_ONLY="${ACUTE_ONLY:-0}"
EXTRA=""
[ "$ACUTE_ONLY" = "1" ] && EXTRA="--acute_only"
# -----------------------------------------------------------------------

echo
echo "=========== DRY RUN ==========="
python "$HOME/convert_aisd.py" --dry_run $EXTRA \
  --dicom_roots "${ROOTS[@]}" \
  --image_root aisd_raw/image --mask_root aisd_raw/mask \
  --raw_root "$nnUNet_raw" --log alignment_dry.csv

echo
read -r -p "Does the summary look right (mostly 'ok', orientation 100% forward)? [y/N] " ans
[ "$ans" = "y" ] || { echo "stopped."; exit 1; }

echo
echo "=========== CONVERT ==========="
python "$HOME/convert_aisd.py" $EXTRA \
  --dicom_roots "${ROOTS[@]}" \
  --image_root aisd_raw/image --mask_root aisd_raw/mask \
  --raw_root "$nnUNet_raw" --log alignment_final.csv

echo
echo "=========== SANITY ==========="
ls "$nnUNet_raw/Dataset402_AISD"
echo "imagesTr: $(ls "$nnUNet_raw/Dataset402_AISD/imagesTr" | wc -l)"
echo "labelsTr: $(ls "$nnUNet_raw/Dataset402_AISD/labelsTr" | wc -l)"
echo "imagesTs: $(ls "$nnUNet_raw/Dataset402_AISD/imagesTs" | wc -l)"
echo "(expect ~340 train / ~52 test)"

echo
echo "=========== PREPROCESS ==========="
rm -rf "$nnUNet_preprocessed/Dataset402_AISD"
LWCTrans_plan_and_preprocess -d 402 -c 3d_fullres --verify_dataset_integrity -np 8

echo
echo "==================================================================="
echo "CHECK THE LOG ABOVE before training:"
echo "  1. reader/writer lines say 'lwctrans.imageio...' NOT 'nnunetv2.imageio...'"
echo "  2. the 3D fullres config shows FLAT keys (conv_kernel_sizes,"
echo "     pool_op_kernel_sizes, UNet_class_name) -- NOT a nested"
echo "     'architecture': {...} block."
echo "Check 2 is the proof that lwctrans's own planner ran. If you see the"
echo "nested block, the patch did not apply and training will fail with"
echo "KeyError: 'conv_kernel_sizes'."
echo
echo "Then: bash ~/05_adaptive_split.sh install"
echo "==================================================================="
