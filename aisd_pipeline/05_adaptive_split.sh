#!/usr/bin/env bash
# LW-CTrans adaptive channel split experiments on AISD (fold 0, 75 epochs; override with LWCTRANS_EPOCHS=N).
#   bash ~/05_adaptive_split.sh install              copy code into the repo and verify it builds
#   bash ~/05_adaptive_split.sh smoke                short end-to-end test of the adaptive variants (about 10 min)
#   bash ~/05_adaptive_split.sh run [variants...]    train, predict the 52 test cases, evaluate, report
#   bash ~/05_adaptive_split.sh report               rebuild the results tables only
# Variants: baseline ratio dynratio gate (default: all four, in that order).
set -uo pipefail

REPO="$HOME/lwctrans-work/LW-CTrans"
DATA="$HOME/lwctrans-work/data"
OUT="$DATA/adaptive_split"
DS="Dataset402_AISD"
OUTPATH="LW-CTrans"
ALL_VARIANTS=(baseline ratio dynratio gate)
SMOKE_VARIANTS=(ratio dynratio gate)
EPOCHS="${LWCTRANS_EPOCHS:-75}"
BASE_TRAINER_PY="$REPO/lwctrans/training/nnUNetTrainer/nnUNetTrainer.py"

die() { echo "ERROR: $*" >&2; exit 1; }
: "${nnUNet_raw:?run: source ~/lwctrans_env.sh}"
: "${nnUNet_preprocessed:?run: source ~/lwctrans_env.sh}"
: "${nnUNet_results:?run: source ~/lwctrans_env.sh}"
TS_IMG="$nnUNet_raw/$DS/imagesTs"
TS_LBL="$nnUNet_raw/$DS/labelsTs"
mkdir -p "$OUT/logs"

trainer_of() {
  case "$1" in
    baseline) echo nnUNetTrainer ;;
    ratio) echo nnUNetTrainerSplitRatio ;;
    dynratio) echo nnUNetTrainerSplitDynRatio ;;
    gate) echo nnUNetTrainerSplitGate ;;
    *) die "unknown variant '$1' (use: ${ALL_VARIANTS[*]})" ;;
  esac
}
model_base() { echo "$nnUNet_results/$DS/$1__nnUNetPlans__3d_fullres"; }
pred_dir() { [ "$1" = baseline ] && echo "$DATA/predictions_test" || echo "$OUT/$1/predictions_test"; }
eval_json() { [ "$1" = baseline ] && echo "$DATA/test_eval_summary.json" || echo "$OUT/$1/test_eval_summary.json"; }

copy_jsons() {
  local base="$1" sub="$2"
  for f in dataset.json plans.json; do
    [ -f "$base/$f" ] || die "missing $base/$f (training did not start correctly)"
    cp "$base/$f" "$base/$sub/"
  done
}

set_epochs() {
  [ -f "$BASE_TRAINER_PY" ] || die "missing $BASE_TRAINER_PY"
  sed -i -E "s/^( {8}self\.num_epochs = )[0-9]+$/\1${EPOCHS}/" "$BASE_TRAINER_PY"
  grep -qE "^ {8}self\.num_epochs = ${EPOCHS}$" "$BASE_TRAINER_PY" || die "could not set num_epochs=${EPOCHS} in $BASE_TRAINER_PY"
  echo ">>> num_epochs = ${EPOCHS} (all four trainers inherit it from nnUNetTrainer.py)"
}

check_epochs() {
  local fold="$1" v="$2" n
  [ -d "$fold" ] || return 0
  if [ -f "$fold/debug.json" ]; then
    n=$(python -c "import json, sys; print(json.load(open(sys.argv[1])).get('num_epochs', ''))" "$fold/debug.json")
  elif [ -f "$fold/validation/summary.json" ] || ls "$fold"/checkpoint_*.pth >/dev/null 2>&1; then
    n="unknown"
  else
    return 0
  fi
  [ "$n" = "$EPOCHS" ] && return 0
  die "$v: $fold holds a run with num_epochs=$n, not $EPOCHS. Delete it and rerun:
    rm -rf \"$(dirname "$fold")\" \"$(pred_dir "$v")\" \"$(eval_json "$v")\""
}

train() {
  local T="$1" sub="$2" log="$3"
  PYTHONUNBUFFERED=1 LWCTrans_train -d 402 -configuration 3d_fullres -fold 0 -outpath "$sub" -tr "$T" --c 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

cmd_install() {
  for f in adaptive_split.py nnUNetTrainerAdaptiveSplit.py adaptive_report.py; do
    [ -f "$HOME/$f" ] || die "upload $f to $HOME first"
  done
  cp "$HOME/adaptive_split.py" "$REPO/lwctrans/network_architecture/adaptive_split.py"
  cp "$HOME/nnUNetTrainerAdaptiveSplit.py" "$REPO/lwctrans/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerAdaptiveSplit.py"
  echo ">>> code copied into $REPO"
  set_epochs

  local ok=1
  grep -q "map_location=self.device, weights_only=False" "$REPO/lwctrans/training/nnUNetTrainer/nnUNetTrainer.py" || { echo "MISSING fix: torch.load weights_only=False in nnUNetTrainer.py"; ok=0; }
  grep -q "map_location=torch.device('cpu'), weights_only=False" "$REPO/lwctrans/inference/predict_from_raw_data.py" || { echo "MISSING fix: torch.load weights_only=False in predict_from_raw_data.py"; ok=0; }
  grep -q "else -1, False)" "$REPO/lwctrans/training/lr_scheduler/polylr.py" && { echo "MISSING fix: PolyLRScheduler verbose arg"; ok=0; }
  [ "$ok" = 1 ] || die "apply the missing fixes with: cd $REPO && python ~/patch_lwctrans.py"
  echo ">>> earlier repo fixes present"

  python - <<'PY' 2>&1 | grep -v "Number of"
import os, torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join
import lwctrans
from lwctrans.utilities.find_class_by_name import recursive_find_python_class
from lwctrans.utilities.plans_handling.plans_handler import PlansManager
pre = join(os.environ['nnUNet_preprocessed'], 'Dataset402_AISD')
pm = PlansManager(load_json(join(pre, 'nnUNetPlans.json')))
cm = pm.get_configuration('3d_fullres')
dsj = load_json(join(pre, 'dataset.json'))
for name in ['nnUNetTrainer', 'nnUNetTrainerSplitRatio', 'nnUNetTrainerSplitDynRatio', 'nnUNetTrainerSplitGate']:
    cls = recursive_find_python_class(join(lwctrans.__path__[0], 'training', 'nnUNetTrainer'), name, 'lwctrans.training.nnUNetTrainer')
    assert cls is not None, f'{name} not found'
    net = cls.build_network_architecture(pm, dsj, cm, 1, enable_deep_supervision=True)
    n = sum(p.numel() for p in net.parameters())
    print(f'  ok  {name:28s} {n:>9d} params ({n / 1e6:.3f} M)')
print(f'  cuda available: {torch.cuda.is_available()}')
PY
  [ "${PIPESTATUS[0]}" = 0 ] || die "network build check failed"
  echo ">>> install OK. Next: bash ~/05_adaptive_split.sh smoke"
}

cmd_smoke() {
  local variants=("$@"); [ ${#variants[@]} -gt 0 ] || variants=("${SMOKE_VARIANTS[@]}")
  local inp="$OUT/smoke_in"
  rm -rf "$inp"; mkdir -p "$inp"
  ls "$TS_IMG"/*_0000.nii.gz | head -2 | while read -r f; do ln -sf "$f" "$inp/"; done
  for v in "${variants[@]}"; do
    [ "$v" = baseline ] && die "smoke supports only: ${SMOKE_VARIANTS[*]}"
    local T base; T=$(trainer_of "$v"); base=$(model_base "$T")
    echo; echo "=========== SMOKE: $v ($T) ==========="
    rm -rf "$base/LW-CTrans-smoke" "$OUT/smoke_pred_$v"
    LWCTRANS_SPLIT_SMOKE=1 train "$T" "LW-CTrans-smoke" "$OUT/logs/smoke_$v.log" || die "smoke training failed for $v (see $OUT/logs/smoke_$v.log)"
    copy_jsons "$base" "LW-CTrans-smoke"
    LWCTrans_predict_from_modelfolder -i "$inp" -o "$OUT/smoke_pred_$v" -m "$base/LW-CTrans-smoke" -f 0 -chk checkpoint_final.pth \
      || die "smoke prediction failed for $v"
    LWCTrans_evaluate_simple "$TS_LBL" "$OUT/smoke_pred_$v" -l 1 -o "$OUT/smoke_pred_$v/eval.json" --chill \
      || die "smoke evaluation failed for $v"
    [ -f "$base/LW-CTrans-smoke/fold_0/split_stats.csv" ] || die "split_stats.csv missing for $v"
    echo ">>> SMOKE PASS: $v"
    rm -rf "$base/LW-CTrans-smoke" "$OUT/smoke_pred_$v"
  done
  rm -rf "$inp"
  echo; echo ">>> all smoke tests passed. Next: bash ~/05_adaptive_split.sh run"
}

cmd_run() {
  local variants=("$@"); [ ${#variants[@]} -gt 0 ] || variants=("${ALL_VARIANTS[@]}")
  local n_img; n_img=$(ls "$TS_IMG"/*.nii.gz | wc -l)
  set_epochs
  for v in "${variants[@]}"; do
    local T base fold pred ej; T=$(trainer_of "$v"); base=$(model_base "$T"); fold="$base/$OUTPATH/fold_0"; pred=$(pred_dir "$v"); ej=$(eval_json "$v")
    echo; echo "=========== VARIANT: $v ($T) ==========="
    check_epochs "$fold" "$v"
    if [ -f "$fold/validation/summary.json" ]; then
      echo ">>> training and validation already done, skipping training"
    else
      echo ">>> training $EPOCHS epochs (resumes automatically if a checkpoint exists). Log: $OUT/logs/train_$v.log"
      train "$T" "$OUTPATH" "$OUT/logs/train_$v.log" || die "training stopped for $v. Run the same command again to resume."
      rm -rf "$pred"
    fi
    copy_jsons "$base" "$OUTPATH"
    mkdir -p "$pred"
    if [ "$(ls "$pred"/*.nii.gz 2>/dev/null | wc -l)" -lt "$n_img" ]; then
      echo ">>> predicting $n_img test cases with checkpoint_best.pth"
      LWCTrans_predict_from_modelfolder -i "$TS_IMG" -o "$pred" -m "$base/$OUTPATH" -f 0 -chk checkpoint_best.pth 2>&1 | tee "$OUT/logs/predict_$v.log"
      [ "${PIPESTATUS[0]}" = 0 ] || die "test prediction failed for $v"
    else
      echo ">>> test predictions already present"
    fi
    echo ">>> evaluating"
    LWCTrans_evaluate_simple "$TS_LBL" "$pred" -l 1 -o "$ej" || die "evaluation failed for $v"
    python -c "import json; d = json.load(open('$ej')); print('>>> $v test Dice:', d['mean']['1']['Dice'])"
  done
  cmd_report
}

cmd_report() {
  echo; echo "=========== REPORT ==========="
  python "$HOME/adaptive_report.py" --data "$DATA" --out "$OUT" --results "$nnUNet_results" --outpath "$OUTPATH"
}

case "${1:-}" in
  install) cmd_install ;;
  smoke) shift; cmd_smoke "$@" ;;
  run) shift; cmd_run "$@" ;;
  report) cmd_report ;;
  *) sed -n '2,7p' "$0"; exit 1 ;;
esac
