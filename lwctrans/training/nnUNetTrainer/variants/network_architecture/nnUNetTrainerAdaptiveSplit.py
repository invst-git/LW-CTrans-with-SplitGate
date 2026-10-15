import csv
import json
import os

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, isfile

from lwctrans.network_architecture.adaptive_split import convert_basic_blocks, adaptive_modules, RatioSplitBlock
from lwctrans.training.lr_scheduler.polylr import PolyLRScheduler
from lwctrans.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PolyLRSchedulerMult(PolyLRScheduler):
    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr * g.get('lr_mult', 1.0)


class _nnUNetTrainerAdaptiveSplit(nnUNetTrainer):
    split_mode = None
    adaptive_lr_mult = float(os.environ.get('LWCTRANS_SPLIT_LR_MULT', '10'))
    tau_anneal_frac = 0.7

    def __init__(self, plans: dict, configuration: str, outpath: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, outpath, fold, dataset_json, unpack_dataset, device)
        self._smoke = os.environ.get('LWCTRANS_SPLIT_SMOKE', '0') == '1'
        if self._smoke:
            self.num_epochs = 1
            self.num_iterations_per_epoch = 10
            self.num_val_iterations_per_epoch = 5
        self._stats = {}

    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json, configuration_manager, num_input_channels,
                                   enable_deep_supervision: bool = True, use_nnunet: bool = False):
        net = nnUNetTrainer.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision, use_nnunet)
        convert_basic_blocks(net, cls.split_mode)
        return net

    def _raw_network(self):
        net = self.network
        for attr in ('module', '_orig_mod'):
            net = getattr(net, attr, net)
        return net

    def do_split(self):
        tr_keys, val_keys = super().do_split()
        if self._smoke:
            val_keys = val_keys[:2]
        return tr_keys, val_keys

    def initialize(self):
        super().initialize()
        n = sum(p.numel() for p in self._raw_network().parameters())
        self.print_to_log_file(f'Adaptive split mode: {self.split_mode}. Total params: {n} ({n / 1e6:.4f} M)')
        for name, m in adaptive_modules(self._raw_network()):
            k = sum(p.numel() for p in m.parameters())
            self.print_to_log_file(f'  {name}: {type(m).__name__} dim={m.dim} params={k}')

    def configure_optimizers(self):
        base, adaptive = [], []
        for name, p in self._raw_network().named_parameters():
            (adaptive if name.endswith('split_logit') or name.endswith('router.2.bias') else base).append(p)
        groups = [{'params': base}]
        if adaptive:
            groups.append({'params': adaptive, 'weight_decay': 0.0, 'lr_mult': self.adaptive_lr_mult})
        optimizer = torch.optim.SGD(groups, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99,
                                    nesterov=True)
        return optimizer, PolyLRSchedulerMult(optimizer, self.initial_lr, self.num_epochs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        frac = self.current_epoch / max(1.0, self.tau_anneal_frac * self.num_epochs)
        for _, m in adaptive_modules(self._raw_network()):
            if isinstance(m, RatioSplitBlock):
                m.set_tau_progress(frac)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._stats = {}

    def validation_step(self, batch: dict) -> dict:
        out = super().validation_step(batch)
        for name, m in adaptive_modules(self._raw_network()):
            if m.last_stat is not None:
                self._stats.setdefault(name, []).append(m.last_stat)
                m.last_stat = None
        return out

    def _aggregate_stats(self):
        row = {}
        for name, vals in self._stats.items():
            v = torch.cat(vals, 0).numpy()
            if v.ndim == 1:
                row[f'{name}_rho_mean'] = float(v.mean())
                row[f'{name}_rho_std'] = float(v.std())
            else:
                per_sample = v.mean(1)
                row[f'{name}_alpha_mean'] = float(v.mean())
                row[f'{name}_alpha_sample_std'] = float(per_sample.std())
                row[f'{name}_alpha_channel_sample_std'] = float(v.std(0).mean())
        for name, m in adaptive_modules(self._raw_network()):
            if isinstance(m, RatioSplitBlock):
                row[f'{name}_tau'] = float(m.tau)
        return row

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        if self.local_rank != 0:
            return
        row = self._aggregate_stats()
        if not row:
            return
        self.print_to_log_file('Split stats: ' + ', '.join(f'{k}={v:.4f}' for k, v in row.items()))
        path = join(self.output_folder, 'split_stats.csv')
        row = {'epoch': self.current_epoch, **row}
        new = not isfile(path)
        with open(path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def on_train_end(self):
        if self.local_rank == 0:
            info = {'split_mode': self.split_mode,
                    'total_params': int(sum(p.numel() for p in self._raw_network().parameters())),
                    'adaptive_lr_mult': self.adaptive_lr_mult,
                    'blocks': {name: m.summary() for name, m in adaptive_modules(self._raw_network())}}
            with open(join(self.output_folder, 'split_final.json'), 'w') as fh:
                json.dump(info, fh, indent=2)
            self.print_to_log_file('Split final: ' + json.dumps(info))
        super().on_train_end()


class nnUNetTrainerSplitRatio(_nnUNetTrainerAdaptiveSplit):
    split_mode = 'ratio'


class nnUNetTrainerSplitDynRatio(_nnUNetTrainerAdaptiveSplit):
    split_mode = 'dynratio'


class nnUNetTrainerSplitGate(_nnUNetTrainerAdaptiveSplit):
    split_mode = 'gate'
