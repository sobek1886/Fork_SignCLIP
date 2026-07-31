"""One-shot dump: ASL-LEX D.Iconicity(M) + 5 PHONO_COLS for exactly the
58-gloss t-SNE probe set (fig:tsne_grid). Stdlib only, no torch/pandas.

Reuses the exact CSV-parsing logic from iconicity_analysis.py::load_asllex
(latin-1 fallback: the published signdata CSV is an Excel export, not valid
UTF-8 -- byte 0xf4 crashed the 2026-07-19 cluster run, and it's not valid
cp1252 either since it also contains 0x8f).

Run on the cluster (where the CSV lives), writes a small JSON to scp back:
    python3 dump_probe58_asllex.py
    -> ~/ASL_Citizen/asllex_probe58.json
"""
import csv
import io
import json
import math
import os

ASLLEX_CSV = os.path.expanduser("~/ASL_Citizen/asllex2_signdata.csv")
OUT = os.path.expanduser("~/ASL_Citizen/asllex_probe58.json")

RATING_DEAF = "D.Iconicity(M)"
PHONO_COLS = ("MajorLocation.2.0", "MinorLocation.2.0", "Handshape.2.0",
              "Movement.2.0", "SignType.2.0")

# gloss -> ASL-LEX Code, resolved locally from
# context/ASL_citizen_context/splits/train.csv (58/58 resolved).
GLOSS2CODE = {
 "AGREEMENT": "C_02_078", "GIRL": "C_03_073", "BOAT": "D_01_055",
 "WIFE": "D_03_025", "CRAZY": "D_03_040", "CAR": "D_02_043",
 "TOOTHBRUSH": "D_03_073", "NIECE": "E_01_080", "ROOF": "F_01_039",
 "MELT": "F_01_050", "FAINT": "G_03_046", "MARCH": "J_01_013",
 "GRANDMOTHER": "C_02_027", "SMOKING": "C_03_087", "OBVIOUS": "E_03_056",
 "FIX": "G_03_043", "UNCLE": "D_02_011", "DIVIDE": "E_02_018",
 "DAUGHTER": "E_02_023", "SAFE": "G_01_085", "BRAINSTORM": "J_01_103",
 "EXPECT": "F_02_068", "WOMAN2": "K_01_101", "EQUAL": "C_01_052",
 "OFFICE": "A_01_063", "DIE": "D_02_021", "TREE": "A_01_002",
 "BOY": "B_02_090", "SISTER": "D_03_050", "QUEEN": "A_03_055",
 "RESCUE": "H_02_067", "MOTHER": "B_02_008", "GRANDFATHER": "B_02_036",
 "BUTTERFLY": "A_03_056", "BROTHER": "B_01_065", "FATHER": "B_01_077",
 "PEACE": "B_01_089", "CAMERA": "B_03_073", "MAN": "C_01_040",
 "SON": "B_02_086", "PARADE": "B_02_039", "FIGHT": "B_03_058",
 "MAKE": "C_01_032", "GET": "B_03_007", "VIOLIN": "A_03_083",
 "MORE": "B_03_032", "WOMAN1": "C_02_028", "KING": "C_01_056",
 "CLEAR": "D_01_006", "HUSBAND": "D_01_082", "HOPE": "D_02_085",
 "FAIR": "G_01_080", "NEPHEW": "A_01_005", "BOOK": "A_01_027",
 "BREAK": "D_01_069", "AUNT": "E_02_027", "SHELF": "E_02_060",
 "SCISSORS": "A_01_038",
}


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
    asllex = load_asllex(ASLLEX_CSV)
    out = {}
    missing = []
    for gloss, code in GLOSS2CODE.items():
        rec = asllex.get(code)
        if rec is None:
            missing.append((gloss, code))
            continue
        out[gloss] = {"code": code, "rating": rec["rating"],
                      "lexical_class": rec["lexical_class"],
                      **{c: rec[c] for c in PHONO_COLS}}
    print(f"resolved {len(out)}/{len(GLOSS2CODE)} probe glosses against ASL-LEX")
    if missing:
        print("MISSING (code not found in ASL-LEX table):", missing)
    n_rated = sum(1 for v in out.values() if v["rating"] is not None)
    print(f"{n_rated}/{len(out)} have a non-null D.Iconicity(M) rating")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
