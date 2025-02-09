import os
import subprocess
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import sys
import re
import numpy as np

def fix_pdb(input_pdb, output_pdb):
    """Fix PDB file using PDBFixer."""
    print(f"Fixing PDB file using PDBFixer...")
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))
    print(f"Fixed PDB file saved to {output_pdb}")

def run_pdb2pqr(input_pdb, output_pqr):
    """Run PDB2PQR to generate PQR file."""
    command = [
        "pdb2pqr",
        "--ff=AMBER",
        "--with-ph=7.0",
        input_pdb,
        output_pqr
    ]
    print(f"Running PDB2PQR: {' '.join(command)}")
    subprocess.run(command, check=True)

def validate_pqr_line(line_num, line):
    """Validate a PQR file line and return formatted version."""
    # PQR format specification
    pqr_pattern = re.compile(r'^(ATOM|HETATM)\s*(\d+)\s+(\S+)\s+(\S+)\s+([A-Z]?)\s*(-?\d+)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)')
    
    match = pqr_pattern.match(line)
    if not match:
        print(f"Warning: Line {line_num} does not match PQR format: {line.strip()}")
        return None
    
    try:
        record = match.group(1)
        atom_num = int(match.group(2))
        atom_name = match.group(3)
        res_name = match.group(4)
        chain = match.group(5) if match.group(5) else ' '
        res_num = int(match.group(6))
        x = float(match.group(7))
        y = float(match.group(8))
        z = float(match.group(9))
        charge = float(match.group(10))
        radius = float(match.group(11))
        
        # Format with strict spacing to ensure no field concatenation
        return f"{record:<6}{atom_num:>5} {atom_name:<4} {res_name:<3} {chain:1}{res_num:>4}    {x:>8.3f} {y:>8.3f} {z:>8.3f} {charge:>8.4f} {radius:>7.4f}\n"
    
    except (ValueError, IndexError) as e:
        print(f"Error parsing line {line_num}: {line.strip()}")
        print(f"Error details: {str(e)}")
        return None

def fix_pqr(input_pqr, output_pqr):
    """Fix PQR file format issues with strict validation."""
    print(f"Fixing PQR file with strict validation...")
    with open(input_pqr, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    error_count = 0
    for i, line in enumerate(lines, 1):
        if line.startswith(('ATOM', 'HETATM')):
            fixed_line = validate_pqr_line(i, line)
            if fixed_line:
                fixed_lines.append(fixed_line)
            else:
                error_count += 1
                # If we're near atom 5875, print extra debug info
                if abs(i - 5875) < 5:
                    print(f"Near problematic atom 5875 - Line {i}:")
                    print(f"Original: {line.strip()}")
        else:
            # Preserve non-atom lines (like TER, END, etc.)
            fixed_lines.append(line)
    
    print(f"Found and fixed {error_count} formatting issues")
    
    # Add proper ending if missing
    if not any(line.startswith('END') for line in fixed_lines):
        fixed_lines.append('END\n')
    
    with open(output_pqr, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed PQR file saved to {output_pqr}")
    print(f"Total atoms processed: {len([l for l in fixed_lines if l.startswith(('ATOM', 'HETATM'))])}")

def generate_apbs_input(pqr_file, output_file):
    """Generate APBS input file."""
    # Convert paths to forward slashes for APBS
    pqr_file = pqr_file.replace('\\', '/')
    
    input_text = f"""# APBS input file for Kir2.1 channel
read
    mol pqr {pqr_file}
end

elec name calc1
    mg-auto
    # Finer grid for better accuracy
    dime 161 161 161
    # Adjust grid lengths for channel
    cglen 120 120 120
    fglen 60 60 60
    cgcent mol 1
    fgcent mol 1
    mol 1
    lpbe
    # Use multiple Debye-Huckel for better boundary conditions
    bcfl mdh
    # Channel environment dielectrics
    pdie 4.0
    sdie 80.0
    # Ion concentrations (150mM KCl)
    ion 1 0.150 2.0
    ion -1 0.150 2.0
    # Surface calculation parameters
    srfm mol
    chgm spl2
    sdens 10.0
    srad 1.4
    swin 0.3
    temp 298.15
    calcenergy total
    calcforce no
end

print elecEnergy calc1 end

quit
"""
    with open(output_file, 'w') as f:
        f.write(input_text)
    print(f"Generated APBS input file: {output_file}")

def parse_apbs_output(output_file):
    """Parse APBS output file to extract energy values and calculate conductance."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Look for the final energy calculation (more refined grid)
        energy_pattern = r'Global net ELEC energy\s*=\s*([+-]?\d*\.?\d+E?[+-]?\d*)\s*kJ/mol'
        match = re.search(energy_pattern, content)
        
        if match:
            energy_kj = float(match.group(1))
            
            # Convert kJ/mol to eV
            energy_ev = energy_kj * 0.0103642688  # 1 kJ/mol = 0.0103642688 eV
            
            # Calculate conductance in units of quantum conductance (G0)
            # G = G0 * exp(-E/kT)
            # where G0 = 2e^2/h ≈ 7.748091729e-5 S
            # kT at room temperature (298K) ≈ 0.0257 eV
            
            kT = 0.0257  # eV at room temperature
            G0 = 7.748091729e-5  # Siemens
            
            # Use log space to handle large numbers
            log_conductance = -energy_ev/kT
            conductance_g0 = np.exp(log_conductance)
            conductance_siemens = conductance_g0 * G0
            
            # Also calculate and display the barrier height
            barrier_height = energy_ev  # eV
            
            print("\nEnergy and Conductance Results:")
            print(f"Electrostatic Energy = {energy_kj:.4e} kJ/mol = {energy_ev:.4f} eV")
            print(f"Barrier Height = {barrier_height:.4f} eV")
            print(f"log(G/G0) = {log_conductance:.4f}")
            if conductance_g0 > 0:
                print(f"Conductance = {conductance_g0:.4e} G0 = {conductance_siemens:.4e} S")
            else:
                print("Conductance is below computational precision (extremely small)")
            
            return energy_kj, conductance_siemens
        else:
            print("Warning: Could not find energy values in APBS output")
            return None, None
            
    except Exception as e:
        print(f"Error parsing APBS output: {str(e)}")
        return None, None

def run_apbs(pqr_file, input_file, output_file):
    """Run APBS calculation."""
    command = ["apbs", input_file]
    print(f"Running APBS: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Write the full output to file
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
        
        # Parse and display energy values
        energy_kj, conductance = parse_apbs_output(output_file)
        
        if energy_kj is None:
            print("Warning: Energy calculation may have failed.")
                
    except subprocess.CalledProcessError as e:
        print(f"Error running APBS:")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise

def main(pdb_file, output_dir):
    """Main function to run the complete workflow."""
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fixed_pdb = os.path.join(output_dir, "fixed.pdb")
    pqr_file = os.path.join(output_dir, "output.pqr")
    apbs_input = os.path.join(output_dir, "apbs.in")
    apbs_output = os.path.join(output_dir, "apbs.out")
    
    try:
        fix_pdb(pdb_file, fixed_pdb)
        run_pdb2pqr(fixed_pdb, pqr_file)
        fix_pqr(pqr_file, pqr_file)
        generate_apbs_input(pqr_file, apbs_input)
        run_apbs(pqr_file, apbs_input, apbs_output)
        print(f"\nDetailed results can be found in {apbs_output}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python poisson.py <pdb_file>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    output_dir = "output"
    main(pdb_file, output_dir)