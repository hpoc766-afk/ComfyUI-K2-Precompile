# SPDX-License-Identifier: GPL-3.0-only
"""Build-manifest lookup for the optional normalized INT4 activation operator."""
import functools
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def library_path():
    path = ROOT / '.kernel-build/norm_quant.json'
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        expected = hashlib.sha256((ROOT / 'kernels/norm_quant.cu').read_bytes()).hexdigest()
        library = ROOT / '.kernel-build' / manifest['library']
        return str(library) if manifest['source_hash'] == expected and library.is_file() else ''
    except (OSError, ValueError, KeyError):
        return ''
