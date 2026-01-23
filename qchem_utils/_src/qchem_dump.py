import json
import pubchempy as pcp
from pathlib import Path


def dump_pubchem_cid(cid: int, outdir="/leonardo_work/EUHPC_A05_006/NeurIS/qchem_utils/pubchem_cache"):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    try:
        compound = pcp.get_compounds(cid, "cid", record_type="3d")[0]
        geom = "3d"
    except Exception:
        compound = pcp.get_compounds(cid, "cid", record_type="2d")[0]
        geom = "2d"

    atoms = []
    for atom in compound.atoms:
        atoms.append(
            {
                "element": atom.element,
                "x": float(atom.x),
                "y": float(atom.y),
                "z": float(atom.z) if geom == "3d" else 0.0,
            }
        )

    data = {
        "cid": cid,
        "geom": geom,
        "formula": compound.molecular_formula,
        "charge": compound.charge,
        # "spin": compound.spin_multiplicity,
        "atoms": atoms,
    }

    outfile = outdir / f"cid_{cid}.json"
    with open(outfile, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {outfile}")
    return outfile


if __name__ == "__main__":
    # dump_pubchem_cid(947)  # N2
    dump_pubchem_cid(62714)