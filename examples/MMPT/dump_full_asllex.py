"""Full-vocabulary ASL-LEX dump: D.Iconicity(M) + 5 PHONO_COLS for every
ASL-Citizen train gloss with an ASL-LEX code (2725 glosses), not just the
58-gloss t-SNE probe set. Used to search for real high-iconicity
phonological triplets across the whole corpus (fig:tsne_grid redesign,
Option 4: extract a new/larger probe set after the 58-gloss set turned up
0 strict 5/5-match triplets).

Stdlib only. Reuses the same latin-1-fallback CSV parsing as
dump_probe58_asllex.py / iconicity_analysis.py::load_asllex.

Run on the cluster:
    python3 dump_full_asllex.py
    -> ~/ASL_Citizen/full_asllex_joined.json
"""
import csv
import io
import json
import math
import os

ASLLEX_CSV = os.path.expanduser("~/ASL_Citizen/asllex2_signdata.csv")
GLOSS2CODE_JSON = os.path.expanduser("~/ASL_Citizen/full_gloss2code.json")
OUT = os.path.expanduser("~/ASL_Citizen/full_asllex_joined.json")

RATING_DEAF = "D.Iconicity(M)"
PHONO_COLS = ("MajorLocation.2.0", "MinorLocation.2.0", "Handshape.2.0",
              "Movement.2.0", "SignType.2.0")


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_asllex(asllex_csv):
    table = {}
    raw = open(asllex_csv, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    with io.StringIO(text, newline="") as f:
        reader = csv.DictReader(f)
        if "Code" not in reader.fieldnames:
            raise ValueError(f"'Code' column missing; got {reader.fieldnames[:8]}...")
        for row in reader:
            code = (row.get("Code") or "").strip()
            if not code:
                continue
            rec = {"rating": _num(row.get(RATING_DEAF)),
                   "lexical_class": (row.get("LexicalClass") or "NA").strip(),
                   "entry_id": (row.get("EntryID") or "").strip()}
            for c in PHONO_COLS:
                rec[c] = (row.get(c) or "NA").strip()
            table[code] = rec
    return table


def main():
    gloss2code = json.load(open(GLOSS2CODE_JSON))
    asllex = load_asllex(ASLLEX_CSV)
    out = {}
    for gloss, code in gloss2code.items():
        rec = asllex.get(code)
        if rec is None:
            continue
        out[gloss] = {"code": code, "rating": rec["rating"],
                      "lexical_class": rec["lexical_class"],
                      **{c: rec[c] for c in PHONO_COLS}}
    n_rated = sum(1 for v in out.values() if v["rating"] is not None)
    print(f"joined {len(out)}/{len(gloss2code)} glosses against ASL-LEX; "
          f"{n_rated} have a non-null D.Iconicity(M) rating")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
