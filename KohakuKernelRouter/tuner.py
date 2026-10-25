# SPDX-License-Identifier: LicenseRef-KohakuMoss-1.0
# Copyright 2026 Shih-Ying Yeh (KohakuBlueLeaf) and contributors.
# Modified for ComfyUI-K2-Precompile: standalone CUDA event timing and table
# selection; trainer/distributed/autograd dependencies removed. Original sources:
# KohakuMossTrainer 0536e62b6062039104d8f9575635b5d07bf573d3,
# src/kohakumoss_kernels/tuning/{timing,tuner,buckets}.py.
"""Kohaku shape-keyed backend selection, settled before graph compilation."""
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import threading
import time

import torch

log = logging.getLogger(__name__)
WARMUP_MS = 25.0


def pow2(n, floor=256):
    b = floor
    while b < n:
        b *= 2
    return b


@contextmanager
def on_device(device):
    if device.type == 'cuda':
        with torch.cuda.device(device):
            yield
    else:
        yield


def _elapsed_ms(fn, calls, device):
    if device.type == 'cpu':
        started = time.perf_counter()
        for _ in range(calls):
            fn()
        return (time.perf_counter() - started) * 1000
    with on_device(device):
        start = torch.Event(device=device, enable_timing=True)
        end = torch.Event(device=device, enable_timing=True)
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)


def warm(fn, device, warmup_ms=WARMUP_MS):
    calls, spent = 2, 0.0
    while spent < warmup_ms:
        spent += _elapsed_ms(fn, calls, device)
        calls *= 2


def best_ms(fn, device, iters=10, flush=None):
    warm(fn, device)
    best = float('inf')
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        best = min(best, _elapsed_ms(fn, 1, device))
    return best


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


class TuneTable:
    """Kohaku's merging per-kernel winner table, with atomic file replacement."""
    def __init__(self, path):
        self.path = Path(path)
        self.rows = read_json(self.path)
        self.lock = threading.RLock()

    def get(self, key):
        row = self.rows.get(key)
        return None if row is None else row['config']

    def put(self, key, config, timings, failures):
        with self.lock:
            self.rows = {**read_json(self.path), **self.rows,
                         key: {'config': config, 'timings_ms': timings, 'failures': failures}}
            temporary = self.path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(self.rows, indent=2, sort_keys=True), encoding='utf-8')
                os.replace(temporary, self.path)
            except OSError as error:
                log.warning('[Kohaku router] cannot persist %s: %s', self.path, error)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def config_name(config):
    return ','.join(f'{k}={v}' for k, v in sorted(config.items()))


def sticky(error):
    text = str(error).lower()
    return any(part in text for part in (
        'illegal memory access', 'device-side assert', 'misaligned address', 'launch failure'))


class Router:
    """Store a winner and bind it once, following Kohaku's fused/split selector.

    Candidates are implementations of the same operator contract. No CUDA Graph
    capture or replay is used. select() is called by the compile backend only;
    native inference receives integer route IDs and never calls this object.
    """
    def __init__(self, directory, device, mode='table'):
        if mode not in ('table', 'off'):
            raise ValueError('K2_KERNEL_TUNE must be table or off')
        self.directory, self.device, self.mode = Path(directory), torch.device(device), mode
        self.tables, self.bound, self.decisions = {}, {}, []
        self.lock = threading.RLock()

    def select(self, kernel, key, candidates, launch):
        with self.lock:
            ident = (kernel, key)
            if ident in self.bound:
                return self.bound[ident]
            table = self.tables.setdefault(kernel, TuneTable(self.directory / f'{kernel}.json'))
            stored = table.get(key)
            if stored in candidates:
                choice, source, times, failures = stored, 'table', {}, {}
            elif self.mode == 'off' or len(candidates) == 1:
                choice, source, times, failures = candidates[0], 'default', {}, {}
            else:
                times, failures, choice = {}, {}, None
                for config in candidates:
                    name = config_name(config)
                    try:
                        ms = best_ms(launch(config), self.device)
                    except Exception as error:
                        if not isinstance(error, (RuntimeError, torch.OutOfMemoryError)) and not type(error).__module__.startswith('triton.'):
                            raise
                        if sticky(error):
                            raise
                        if isinstance(error, torch.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        failures[name] = str(error)[:500]
                        continue
                    times[name] = ms
                    if choice is None or ms < times[config_name(choice)]:
                        choice = config
                if choice is None:
                    raise RuntimeError(f'Kohaku router: every {kernel} candidate failed: {failures}')
                source = 'measured'
                table.put(key, choice, times, failures)
            self.bound[ident] = choice
            decision = {'kernel': kernel, 'key': key, 'config': choice, 'source': source,
                        'timings_ms': times, 'failures': failures}
            self.decisions.append(decision)
            log.info('[Kohaku router] %s %s: %s (%s), %s',
                     kernel, key, config_name(choice), source, times)
            return choice
