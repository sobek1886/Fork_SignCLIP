#!/usr/bin/env python3
"""Diagnose ngt_pair_manifest.json path issues."""
import json
import os
import sys

manifest_path = os.path.expanduser("~/ngt_pair_manifest.json")
if len(sys.argv) > 1:
    manifest_path = sys.argv[1]

with open(manifest_path) as f:
    manifest = json.load(f)

print(f"Total signs: {len(manifest)}")

first_key = list(manifest.keys())[0]
first_val = manifest[first_key]
print(f"\nFirst sign_id: {first_key}")
print(f"Value type: {type(first_val).__name__}")

if isinstance(first_val, dict):
    print(f"Keys: {list(first_val.keys())}")
    for k, paths in first_val.items():
        print(f"\n  [{k}] ({len(paths)} paths)")
        for p in paths[:2]:
            exists = os.path.exists(p)
            print(f"    {'OK  ' if exists else 'MISS'} {p}")
        if len(paths) > 2:
            print(f"    ... {len(paths) - 2} more")
elif isinstance(first_val, list):
    print(f"Flat list with {len(first_val)} paths:")
    for p in first_val[:3]:
        exists = os.path.exists(p)
        print(f"  {'OK  ' if exists else 'MISS'} {p}")

# Summary: count missing files
print("\n--- Summary ---")
missing = 0
total = 0
for sign_id, val in manifest.items():
    if isinstance(val, dict):
        all_paths = val.get('real', []) + val.get('unreal', [])
    else:
        all_paths = val
    for p in all_paths:
        total += 1
        if not os.path.exists(p):
            missing += 1

print(f"Total paths: {total}")
print(f"Missing:     {missing}")
print(f"OK:          {total - missing}")
