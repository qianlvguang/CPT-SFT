#!/usr/bin/env python3
"""Check a target regex against actual module names collected on the target job."""
import argparse
import re
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pattern", help="Python regular expression, applied with re.fullmatch")
parser.add_argument("names_file", help="UTF-8 file containing one module name per line")
args = parser.parse_args()
try:
    pattern = re.compile(args.pattern)
    with open(args.names_file, encoding="utf-8") as names_file:
        names = [line.strip() for line in names_file if line.strip()]
except (OSError, re.error) as exc:
    parser.error(str(exc))
matched = [name for name in names if pattern.fullmatch(name)]
for name in matched:
    print(name)
print(f"matched {len(matched)} of {len(names)} names", file=sys.stderr)
if not matched:
    sys.exit(1)
