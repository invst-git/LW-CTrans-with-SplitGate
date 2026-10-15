#!/usr/bin/env bash
# AISD data download. Run AFTER 01_setup.sh and `source ~/lwctrans_env.sh`.
#   bash 02_get_data.sh
#
# Known from the previous run: dicom-1 and dicom-3 failed gzip integrity on
# every retry (likely bad at the source, not a transient network fault).
# dicom-1 still yields ~68 patients via partial extraction, which is why we
# extract corrupt archives anyway instead of deleting them. Coverage from
# dicom-0 (221) + dicom-1 partial (68) + dicom-2 (110) = 399 -- effectively
# the whole dataset, so dicom-3 is not worth chasing.
set -uo pipefail   # deliberately NOT -e: corrupt archives are expected

DATA="$HOME/lwctrans-work/data"
cd "$DATA"
mkdir -p aisd_raw aisd_raw_dicom

# Google Drive file IDs from the AISD README. These worked previously; if any
# download 404s or returns an HTML page, re-check the README for updated links.
declare -A FILES=(
  [image.zip]=157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk
  [mask.zip]=1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6
  [dicom-0.tar.gz]=1WZdjvq0ZFPnNpbhfxadIYPcbWpdjgM-n
  [dicom-1.tar.gz]=1CBVHFq8m3-FjG3-aCRfkUMzgHsfheTLd
  [dicom-2.tar.gz]=1ZBK93GfTwpfJTv5DlD2vhZnbO5JXpwrz
)

for f in "${!FILES[@]}"; do
  if [ -f "$f" ]; then echo ">>> $f already present, skipping download"; continue; fi
  echo ">>> downloading $f"
  gdown "${FILES[$f]}" -O "$f"
done

echo
echo "=========== integrity ==========="
for f in dicom-0.tar.gz dicom-1.tar.gz dicom-2.tar.gz; do
  [ -f "$f" ] || continue
  if gzip -t "$f" 2>/dev/null; then echo "  CLEAN   $f"; else echo "  CORRUPT $f (will salvage what extracts)"; fi
done

echo
echo "=========== extract PNGs ==========="
[ -d aisd_raw/image ] || unzip -q image.zip -d aisd_raw
[ -d aisd_raw/mask ]  || unzip -q mask.zip  -d aisd_raw

echo
echo "=========== extract DICOM ==========="
# tar aborts at the corruption point but keeps everything before it -- that
# partial content is exactly how we recover ~68 patients from dicom-1.
for f in dicom-0 dicom-1 dicom-2; do
  [ -f "$f.tar.gz" ] || continue
  [ -d "aisd_raw_dicom/$f" ] && { echo "  $f already extracted"; continue; }
  echo "  extracting $f ..."
  tar xzf "$f.tar.gz" -C aisd_raw_dicom 2>/dev/null || echo "    (aborted early -- partial extraction kept)"
done

echo
echo "=========== coverage ==========="
TOTAL=0
for d in aisd_raw_dicom/dicom-*; do
  [ -d "$d" ] || continue
  n=$(ls "$d" 2>/dev/null | wc -l)
  echo "  $(basename "$d"): $n patients"
  TOTAL=$((TOTAL + n))
done
echo "  ---"
echo "  DICOM patients total: $TOTAL"
echo "  PNG image folders:    $(ls aisd_raw/image 2>/dev/null | wc -l)"
echo "  PNG mask folders:     $(ls aisd_raw/mask 2>/dev/null | wc -l)"
echo
echo "Expected roughly: dicom-0 221, dicom-1 ~68 (partial), dicom-2 110; PNG 398."
echo "Next: bash 03_convert.sh"
