import os
import sys
import subprocess
import numpy as np
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def fix_pdb(pdb_file, fixed_pdb_file):
    """Fix the PDB using PDBFixer."""
    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    with open(fixed_pdb_file, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

def run_pdb2pqr(pdb_in, pqr_out, force_field="AMBER", ph=7.0):
    """Convert PDB → PQR using PDB2PQR."""
    cmd = [
        "pdb2pqr",
        f"--ff={force_field}",
        f"--with-ph={ph}",
        "--keep-chain",     # Keep chain IDs
        pdb_in,
        pqr_out
    ]
    print("[INFO] Running PDB2PQR:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_pnp(pqr_file, apbs_input, dx_prefix="pnp_pot"):
    """
    Run a Poisson–Nernst–Planck (PNP) calculation
    using a specialized APBS build that supports 'pnp' blocks.
    """
    with open(apbs_input, "w") as f:
        f.write(f"""read
    mol pqr {pqr_file}
end

# PNP block:
pnp name pnpcalc
    dime 129 129 129
    # Grid lengths (adjust to your system size!)
    cglen 40.0 40.0 40.0
    fglen 30.0 30.0 30.0
    
    # Centering on molecule
    cgcent mol 1
    fgcent mol 1

    # Dielectric settings
    pdie 2.0
    sdie 78.0
    bcfl sdh

    # Ion parameters (example for monovalent + and -)
    ion charge  1.0  conc 0.150  radius 2.0
    ion charge -1.0  conc 0.150  radius 2.0

    # Additional PNP parameters would go here:
    # iteration settings, tolerance, etc. ...
    
    mol 1
end

# Print potential in .dx format
print grid pnpcalc format dx prefix {dx_prefix}

quit
""")
    print(f"[INFO] PNP input file generated: {apbs_input}")

    cmd = ["apbs", apbs_input]
    print("[INFO] Running APBS-PNP:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_dx(dx_file):
    """Example of parsing a .dx file into a 1D numpy array."""
    data = []
    with open(dx_file) as f:
        in_data = False
        for line in f:
            if line.startswith("object 3 class array"):
                in_data = True
                continue
            if in_data:
                # Attempt to parse floats
                parts = line.strip().split()
                try:
                    floats = list(map(float, parts))
                    data.extend(floats)
                except ValueError:
                    break
    return np.array(data)

def main(pdb_file, out_dir="pnp_output"):
    # Make output directory
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Fix PDB
    fixed_pdb = os.path.join(out_dir, "fixed.pdb")
    fix_pdb(pdb_file, fixed_pdb)

    # Step 2: Generate PQR
    pqr_file = os.path.join(out_dir, "output.pqr")
    run_pdb2pqr(fixed_pdb, pqr_file)

    # Step 3: Run PNP solver
    apbs_in = os.path.join(out_dir, "pnp.in")
    dx_prefix = "pnp_pot"
    run_pnp(pqr_file, apbs_in, dx_prefix)

    # Step 4: Parse potential
    dx_file = os.path.join(out_dir, f"{dx_prefix}.dx")
    if not os.path.isfile(dx_file):
        print("[ERROR] PNP .dx output not found!")
        return

    pot_array = parse_dx(dx_file)
    print(f"[INFO] PNP potential array length = {len(pot_array)}")

    # Step 5: (Optional) Some toy analysis ...
    # Real PNP outputs might include flux files or local concentrations
    # depending on advanced input settings. Then you'd do conduction analysis.

    print("[DONE] Poisson–Nernst–Planck pipeline complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python poisson_pnp.py <input_pdb> [output_dir]")
        sys.exit(1)
    pdb_in = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "pnp_output"
    main(pdb_in, output_folder)
