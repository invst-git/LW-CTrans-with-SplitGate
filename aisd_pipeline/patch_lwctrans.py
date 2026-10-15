#!/usr/bin/env python3
"""Consolidated one-time patch for the LW-CTrans repo.

Fixes three independent classes of bug found while getting this repo to
run. All are latent in the upstream repo -- none are caused by your setup.

  BUG 1 (9 files) -- trainer variants missing `outpath`.
      LW-CTrans added an `outpath` parameter to its base nnUNetTrainer,
      and run_training.py always passes outpath=... to whichever trainer
      class is selected. But every variant subclass was copied from
      upstream nnU-Net and still has the old signature, so any `-tr`
      other than the plain nnUNetTrainer dies with:
        TypeError: <Variant>.__init__() got an unexpected keyword argument 'outpath'

  BUG 2 (7 files) -- hardcoded `nnunetv2.*` module-lookup prefixes.
      The repo resolves several classes by NAME at runtime via
      recursive_find_python_class(folder, name, current_module). It
      searches the lwctrans/ folder correctly but then IMPORTS using the
      leftover prefix "nnunetv2.X". If vanilla nnunetv2 happens to be
      installed, this silently loads ITS classes instead of lwctrans's
      own -- including its ExperimentPlanner, which writes a
      newer-schema plans file that lwctrans's own ConfigurationManager
      cannot read:
        KeyError: 'conv_kernel_sizes'
      If vanilla nnunetv2 is NOT installed, the same lines fail with
      ImportError instead. Patching fixes both cases.

  BUG 3 (1 file) -- PolyLRScheduler passes the removed `verbose` arg.
      Modern PyTorch dropped `verbose` from LRScheduler.__init__:
        TypeError: LRScheduler.__init__() takes from 2 to 3 positional
                   arguments but 4 were given
      Dropping the argument is safe on old torch too -- False was
      already the default.

  BUG 4 (3 files) -- torch.load() hits the new weights_only=True default.
      PyTorch 2.6 flipped torch.load's default from weights_only=False to
      True, which blocks unpickling of some ordinary checkpoint contents
      (here, a numpy scalar type) as an anti-RCE measure for untrusted
      files:
        _pickle.UnpicklingError: Weights only load failed ...
      This bites on `--c` (resume) or any inference run, not on a fresh
      training start (which never calls torch.load). Since these are
      your own checkpoints, not untrusted downloads, explicit
      weights_only=False is safe and correct here.

Idempotent: safe to run twice; already-patched files are skipped.

Usage:
    cd ~/lwctrans-work/LW-CTrans
    python patch_lwctrans.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------- BUG 1
OUTPATH_FILES = [
    "lwctrans/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py",
    "lwctrans/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs_noDataLoading.py",
    "lwctrans/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDA5.py",
    "lwctrans/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdan.py",
    "lwctrans/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdam.py",
    "lwctrans/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py",
    "lwctrans/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py",
    "lwctrans/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py",
    "lwctrans/training/nnUNetTrainer/variants/sampling/nnUNetTrainer_probabilisticOversampling.py",
]

OUTPATH_SUBS = [
    ("def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, "
     "unpack_dataset: bool = True,",
     "def __init__(self, plans: dict, configuration: str, outpath: str, fold: int, dataset_json: dict, "
     "unpack_dataset: bool = True,"),
    ("super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)",
     "super().__init__(plans, configuration, outpath, fold, dataset_json, unpack_dataset, device)"),
]

# ---------------------------------------------------------------- BUG 2
LOOKUP_SUBS = [
    ("lwctrans/experiment_planning/plan_and_preprocess_api.py",
     'current_module="nnunetv2.experiment_planning")',
     'current_module="lwctrans.experiment_planning")'),
    ("lwctrans/utilities/label_handling/label_handling.py",
     'current_module="nnunetv2.utilities.label_handling")',
     'current_module="lwctrans.utilities.label_handling")'),
    ("lwctrans/utilities/plans_handling/plans_handler.py",
     'current_module="nnunetv2.preprocessing")',
     'current_module="lwctrans.preprocessing")'),
    ("lwctrans/utilities/plans_handling/plans_handler.py",
     'current_module="nnunetv2.experiment_planning")',
     'current_module="lwctrans.experiment_planning")'),
    ("lwctrans/preprocessing/preprocessors/default_preprocessor.py",
     "'nnunetv2.preprocessing.normalization')",
     "'lwctrans.preprocessing.normalization')"),
    ("lwctrans/preprocessing/resampling/utils.py",
     "'nnunetv2.preprocessing.resampling')",
     "'lwctrans.preprocessing.resampling')"),
    ("lwctrans/imageio/reader_writer_registry.py",
     "'nnunetv2.imageio')",
     "'lwctrans.imageio')"),
    ("lwctrans/inference/predict_from_raw_data.py",
     "'nnunetv2.training.nnUNetTrainer')",
     "'lwctrans.training.nnUNetTrainer')"),
]

# ---------------------------------------------------------------- BUG 3
POLYLR_FILE = "lwctrans/training/lr_scheduler/polylr.py"
POLYLR_OLD = "super().__init__(optimizer, current_step if current_step is not None else -1, False)"
POLYLR_NEW = "super().__init__(optimizer, current_step if current_step is not None else -1)"

# ---------------------------------------------------------------- BUG 4
TORCH_LOAD_SUBS = [
    ("lwctrans/training/nnUNetTrainer/nnUNetTrainer.py",
     "torch.load(filename_or_checkpoint, map_location=self.device)",
     "torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)"),
    ("lwctrans/inference/predict_from_raw_data.py",
     "torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),\n"
     "                                    map_location=torch.device('cpu'))",
     "torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),\n"
     "                                    map_location=torch.device('cpu'), weights_only=False)"),
    ("lwctrans/run/load_pretrained_weights.py",
     "torch.load(fname)",
     "torch.load(fname, weights_only=False)"),
]

# ---------------------------------------------------------------- paths
PATHS_FILE = "lwctrans/paths.py"
PATHS_SUBS = [
    ("os.environ['nnUNet_raw'] = '/home/wangyh/Codes/nnUNetv2/nnunetv2/data/nnUNet_raw'", ""),
    ("os.environ['nnUNet_preprocessed'] = '/home/wangyh/Codes/nnUNetv2/nnunetv2/data/nnUNet_preprocessed'", ""),
    ("os.environ['nnUNet_results'] = '/home/wangyh/Codes/LW-CTrans/results'", ""),
]


def apply(path: Path, subs, label):
    """Apply (old, new) pairs to one file. Returns number of files changed."""
    if not path.exists():
        print(f"  MISSING (skipped): {path}")
        return 0
    text = original = path.read_text()
    hits = 0
    for old, new in subs:
        n = text.count(old)
        if n:
            text = text.replace(old, new)
            hits += n
    if text != original:
        path.write_text(text)
        print(f"  patched [{label}] {path}  ({hits} replacement(s))")
        return 1
    print(f"  already ok [{label}] {path}")
    return 0


def main() -> None:
    root = Path.cwd()
    if not (root / "lwctrans").is_dir() or not (root / "setup.py").exists():
        sys.exit("ERROR: run this from the LW-CTrans repo root (the folder with setup.py).")

    changed = 0

    print("BUG 1: trainer variants missing 'outpath'")
    for rel in OUTPATH_FILES:
        changed += apply(root / rel, OUTPATH_SUBS, "outpath")

    print("\nBUG 2: hardcoded nnunetv2.* module-lookup prefixes")
    for rel, old, new in LOOKUP_SUBS:
        changed += apply(root / rel, [(old, new)], "lookup")

    print("\nBUG 3: PolyLRScheduler passes removed 'verbose' arg")
    changed += apply(root / POLYLR_FILE, [(POLYLR_OLD, POLYLR_NEW)], "polylr")

    print("\nBUG 4: torch.load() hits the new weights_only=True default")
    for rel, old, new in TORCH_LOAD_SUBS:
        changed += apply(root / rel, [(old, new)], "torchload")

    print("\nEXTRA: neutralise hardcoded author paths in paths.py")
    changed += apply(root / PATHS_FILE, PATHS_SUBS, "paths")

    print(f"\n=== {changed} file(s) modified ===")

    # verification pass
    print("\nVerification:")
    bad = []
    for rel in OUTPATH_FILES:
        t = (root / rel).read_text() if (root / rel).exists() else ""
        if "super().__init__(plans, configuration, fold," in t:
            bad.append(f"outpath not fixed: {rel}")
    for rel, old, _ in LOOKUP_SUBS:
        t = (root / rel).read_text() if (root / rel).exists() else ""
        if old in t:
            bad.append(f"lookup not fixed: {rel}")
    t = (root / POLYLR_FILE).read_text()
    if POLYLR_OLD in t:
        bad.append("polylr not fixed")
    for rel, old, _ in TORCH_LOAD_SUBS:
        t = (root / rel).read_text() if (root / rel).exists() else ""
        if old in t:
            bad.append(f"torch.load not fixed: {rel}")
    t = (root / PATHS_FILE).read_text()
    if "/home/wangyh/" in t:
        bad.append("paths.py still contains hardcoded author paths")

    if bad:
        print("  PROBLEMS FOUND:")
        for b in bad:
            print(f"    - {b}")
        sys.exit(1)
    print("  all checks passed")

    # syntax check every touched file
    import ast
    for rel in OUTPATH_FILES + [s[0] for s in LOOKUP_SUBS] + [POLYLR_FILE, PATHS_FILE] + [s[0] for s in TORCH_LOAD_SUBS]:
        p = root / rel
        if p.exists():
            try:
                ast.parse(p.read_text())
            except SyntaxError as e:
                sys.exit(f"SYNTAX ERROR in {rel}: {e}")
    print("  all patched files parse cleanly")


if __name__ == "__main__":
    main()
