from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, e, k, nano


# --- Step 1: Set Up Molecular Dynamics Simulation ---
def run_md_simulation(pdb_file, output_trajectory="trajectory.dcd", output_pdb="output.pdb"):
    """
    Run Molecular Dynamics Simulation using OpenMM.
    """
    # Load the PDB structure
    pdb = PDBFile(pdb_file)

    # Define force field
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Create a solvated system with explicit water and ions
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer, ionicStrength=0.15*molar)

    # Create the OpenMM system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0*nanometer,
        constraints=HBonds
    )

    # Set up the integrator and simulation
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Minimize energy
    print("Minimizing energy...")
    simulation.minimizeEnergy()

    # Equilibrate the system
    print("Equilibrating...")
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    simulation.reporters.append(DCDReporter(output_trajectory, 1000))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    simulation.step(5000)  # Equilibration steps

    # Run production MD
    print("Running production MD...")
    simulation.reporters.append(DCDReporter(output_trajectory, 1000))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    simulation.step(50000)  # Production steps

    # Save the final structure
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(output_pdb, 'w'))
    print(f"Simulation complete. Results saved to {output_trajectory} and {output_pdb}.")


# --- Step 2: Analyze the MD Trajectory ---
def analyze_md_trajectory(trajectory_file, topology_file, z_min, z_max, pore_radius):
    """
    Analyze the MD trajectory using MDAnalysis.
    """
    u = mda.Universe(topology_file, trajectory_file)

    # Analyze the pore properties from the trajectory
    pore_atoms = u.select_atoms(f"prop z > {z_min} and prop z < {z_max} and around {pore_radius} (protein)")
    print(f"Number of selected atoms in pore region: {len(pore_atoms)}")

    # Return the analyzed data
    return pore_atoms


# --- Step 3: Calculate Charge Density ---
def calculate_charge_density(pore_atoms, z_min, z_max, pore_radius):
    """
    Calculate the charge density in the pore region.
    """
    # Define residue charges (include only charged residues)
    residue_charges = {
        'ASP': -1.0,  # Aspartate
        'GLU': -1.0,  # Glutamate
        'LYS': 1.0,   # Lysine
        'ARG': 1.0    # Arginine
    }

    # Calculate total charge
    total_charge = 0
    for residue in pore_atoms.residues:
        if residue.resname in residue_charges:
            total_charge += residue_charges[residue.resname]

    # Calculate pore volume
    pore_height = z_max - z_min
    pore_volume = np.pi * (pore_radius**2) * pore_height  # Volume in nm^3
    pore_volume_m3 = pore_volume * 1e-27  # Convert nm^3 to m^3

    # Charge density
    charge_density = (total_charge * e) / pore_volume_m3 if total_charge != 0 else 0.0
    return charge_density


# --- Step 4: Main Function ---
def main():
    pdb_file = "7zdznew.pdb"  # Input PDB file
    trajectory_file = "trajectory.dcd"
    output_pdb = "output.pdb"

    # Step 1: Run MD simulation
    run_md_simulation(pdb_file, trajectory_file, output_pdb)

    # Step 2: Analyze MD trajectory
    z_min, z_max = 139.68, 274.70
    pore_radius = 2.0
    pore_atoms = analyze_md_trajectory(trajectory_file, output_pdb, z_min, z_max, pore_radius)

    # Step 3: Calculate charge density
    charge_density = calculate_charge_density(pore_atoms, z_min, z_max, pore_radius)
    print(f"Charge Density: {charge_density:.2e} C/m^3")


# --- Run the Script ---
if __name__ == "__main__":
    main()
