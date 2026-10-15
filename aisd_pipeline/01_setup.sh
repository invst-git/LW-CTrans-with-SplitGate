#!/usr/bin/env bash
# LW-CTrans + AISD :: environment setup for a fresh qBraid A100 GPU instance.
# Run once:  bash 01_setup.sh
set -euo pipefail

WORK="$HOME/lwctrans-work"
VENV="$HOME/envs/lwctrans"
DATA="$WORK/data"

echo "=========== GPU / driver ==========="
nvidia-smi
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo ">>> driver supports CUDA ${DRIVER_CUDA}"
echo ">>> this script installs the cu128 torch build (matches driver 570.x / CUDA 12.8)"
echo ">>> if the line above shows something OTHER than 12.8, STOP and adjust TORCH_INDEX below"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo
echo "=========== folders ==========="
mkdir -p "$WORK" "$DATA" "$DATA/nnUNet_raw" "$DATA/nnUNet_preprocessed" "$DATA/nnUNet_results"

echo
echo "=========== repo ==========="
cd "$WORK"
[ -d LW-CTrans ] || git clone https://github.com/hulinkuang/LW-CTrans.git

echo
echo "=========== venv ==========="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools --quiet

echo
echo "=========== deps ==========="
# batchgenerators (2023-era) predates NumPy 2.
python -m pip install "numpy<2" --quiet
# Imported by the code but MISSING from the repo's setup.py:
#   thop   -> get_network_from_plans.py (module-level import, breaks immediately)
#   einops -> Coformer attention blocks
#   timm   -> DropPath in light_transformer_block.py
#   medpy  -> validation metrics
python -m pip install thop einops timm medpy --quiet
# Tooling for data prep.
python -m pip install gdown pydicom SimpleITK pillow matplotlib --quiet

echo
echo "=========== repo install ==========="
cd "$WORK/LW-CTrans"
python -m pip install -e . --quiet

echo
echo "=========== torch LAST (critical) ==========="
# setup.py only pins torch>=2.0.0 with no CUDA constraint, so `pip install -e .`
# will happily pull a default-index build compiled for a CUDA newer than the
# driver. Installing torch AFTER the repo, with --force-reinstall and an
# explicit index, is what makes this stick. torch/torchvision/torchaudio ship
# as a matched trio -- installing torch alone leaves torchvision compiled
# against the old ABI ("operator torchvision::nms does not exist").
python -m pip install --force-reinstall --index-url "$TORCH_INDEX" \
    torch torchvision torchaudio

echo
echo "=========== patch repo ==========="
if [ -f "$HOME/patch_lwctrans.py" ]; then
    python "$HOME/patch_lwctrans.py"
else
    echo "!! $HOME/patch_lwctrans.py not found -- upload it, then run:"
    echo "   cd $WORK/LW-CTrans && python ~/patch_lwctrans.py"
fi

echo
echo "=========== env vars ==========="
cat > "$HOME/lwctrans_env.sh" <<EOF
source $VENV/bin/activate
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$DATA/nnUNet_results"
export nnUNet_def_n_proc=8
EOF

# Only append to .bashrc once.
if ! grep -q "lwctrans_env.sh" "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# ---- LW-CTrans ----\nsource %s/lwctrans_env.sh\n' "$HOME" >> "$HOME/.bashrc"
fi

echo
echo "=========== verify ==========="
python - <<'PY'
import torch, torchvision, numpy
print(f"torch        {torch.__version__}  (built for CUDA {torch.version.cuda})")
print(f"torchvision  {torchvision.__version__}")
print(f"numpy        {numpy.__version__}")
ok = torch.cuda.is_available()
print(f"cuda available: {ok}")
if ok:
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("!! CUDA NOT AVAILABLE -- torch/driver mismatch, stop and fix before continuing")
PY

for m in thop einops timm medpy pydicom SimpleITK; do
    python -c "import $m" 2>/dev/null && echo "  ok  $m" || echo "  FAIL $m"
done

echo
echo "==================================================================="
echo "Setup done. IMPORTANT: this shell does not have the env vars yet."
echo "Run this now, and at the start of every NEW terminal:"
echo "    source ~/lwctrans_env.sh"
echo "Then continue with 02_get_data.sh"
echo "==================================================================="
