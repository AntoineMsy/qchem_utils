import json
from pathlib import Path


class CachedAtom:
    def __init__(self, element, x, y, z):
        self.element = element
        self.x = x
        self.y = y
        self.z = z


class CachedCompound:
    def __init__(self, data):
        self.cid = data["cid"]
        self.molecular_formula = data.get("formula", None)
        self.charge = data.get("charge", 0)
        self.spin_multiplicity = data.get("spin", None)
        self.geom = data["geom"]

        self.atoms = [
            CachedAtom(a["element"], a["x"], a["y"], a["z"])
            for a in data["atoms"]
        ]


def load_cached_compound(cid: int, cachedir="pubchem_cache"):
    path = Path(cachedir) / f"cid_{cid}.json"
    if not path.exists():
        raise FileNotFoundError(f"No cached PubChem file for CID {cid}")

    with open(path) as f:
        data = json.load(f)

    return CachedCompound(data)