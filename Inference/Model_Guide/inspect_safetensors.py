#!/usr/bin/env python3
"""Read safetensors headers without loading tensor payloads."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import struct
import sys


def category(name):
    if '.ple.' in name or 'ngram_embedding' in name:
        return 'ple'
    if 'visual.merger' in name:
        return 'aligner'
    if '.visual.' in name or name.startswith('visual.'):
        return 'vision'
    if '.shared_expert' in name:
        return 'shared_expert_and_gate'
    if re.search(r'\.mlp\.(gate|router)(\.|$)', name):
        return 'router'
    if '.experts.' in name or name.endswith('.experts'):
        return 'routed_experts'
    if name.startswith('lm_head'):
        return 'lm_head'
    if 'mtp' in name:
        return 'mtp'
    if 'language_model' in name:
        return 'language_model_other'
    return 'other'


def read_header(path):
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('missing header length')
        length = struct.unpack('<Q', raw)[0]
        if length == 0 or length > 100_000_000 or 8 + length > path.stat().st_size:
            raise ValueError(f'invalid header length: {length}')
        header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError('header is not a JSON object')
        return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--filter', default='.', help='Python regex over tensor names')
    parser.add_argument('--list', action='store_true', help='print individual matching tensors')
    args = parser.parse_args()
    try:
        pattern = re.compile(args.filter)
    except re.error as exc:
        parser.error(str(exc))
    files = sorted(args.model_dir.glob('*.safetensors'))
    if not files:
        parser.error('no .safetensors files found in directory')
    totals = defaultdict(lambda: [0, 0])
    try:
        for path in files:
            header = read_header(path)
            for name, info in header.items():
                if name == '__metadata__' or not pattern.search(name):
                    continue
                start, end = info['data_offsets']
                if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end):
                    raise ValueError(f'invalid offsets for {name}')
                size = end - start
                group = category(name)
                dtype = info['dtype']
                totals[(group, dtype)][0] += 1
                totals[(group, dtype)][1] += size
                if args.list:
                    print(f'{group}\t{dtype}\t{info["shape"]}\t{size}\t{path.name}\t{name}')
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'Invalid or unreadable safetensors header: {exc}', file=sys.stderr)
        return 1
    print(f'Scanned {len(files)} local shard(s); completeness is not verified.', file=sys.stderr)
    print('category\tdtype\ttensors\tdata_bytes\tGiB')
    for (group, dtype), (count, size) in sorted(totals.items()):
        print(f'{group}\t{dtype}\t{count}\t{size}\t{size / 2**30:.4f}')
    if not totals:
        print('No tensors matched.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
