#!/usr/bin/env python3
"""
Neuro Ion Channel Conductance Calculator
===========================================

This script prepares a membrane-embedded ion channel system (via tleap),
runs an MD simulation with an applied electric field (using OpenMM) to drive ion permeation,
and then analyzes the trajectory (using MDAnalysis) to count ion crossing events.
The ionic current is computed from the net number of crossing events, and the conductance is obtained by I/V.

Usage:
    python ionchannel_md.py input_structure.pdb --voltage 0.1 --time_ns 20 --ion K+

Arguments:
    input_pdb   : The PDB file of your membrane-embedded ion channel.
    --voltage   : The applied voltage in volts.
    --time_ns   : Simulation time in nanoseconds (default: 20 ns).
    --ion       : Ion type to analyze (choices: K+, Na+, Cl-).

Note:
    This script uses tleap (from AmberTools) to prepare the system. Ensure that tleap is installed and in your PATH.
"""

import os
import sys
import subprocess
import argparse
import numpy as np

# OpenMM and OpenMM App imports
from openmm import unit, Platform, LangevinIntegrator, Vec3, CustomExternalForce, NonbondedForce
from openmm.app import Simulation, DCDReporter, PDBFile, StateDataReporter, PME

# MDTraj is imported below if needed; MDAnalysis is used for analysis.
import MDAnalysis as mda

# ------------------------------------------------------------------------------
# Ion parameters (Amber style)
ION_PARAMS = {
    'K+': {
        'resname': 'K',
        'charge': 1.0 * unit.elementary_charge,
        'sigma': 0.303763 * unit.nanometer,
        'epsilon': 0.36923 * unit.kilojoule_per_mole,
    },
    'Na+': {
        'resname': 'Na',
        'charge': 1.0 * unit.elementary_charge,
        'sigma': 0.212557 * unit.nanometer,
        'epsilon': 0.0874393 * unit.kilojoule_per_mole,
    },
    'Cl-': {
        'resname': 'Cl',
        'charge': -1.0 * unit.elementary_charge,
        'sigma': 0.440104 * unit.nanometer,
        'epsilon': 0.41840 * unit.kilojoule_per_mole,
    }
}

# ------------------------------------------------------------------------------
# System Preparation Functions

def create_tleap_script(pdb_file, output_prefix="system"):
    """Creates a tleap input script to generate Amber topology/coordinate files."""
    script = f"""
# Load force fields
source leaprc.protein.ff14SB
source leaprc.water.tip3p

# Load PDB file
mol = loadpdb {pdb_file}

# Add missing atoms and hydrogens; add neutralizing ions (dummy call; adjust as needed)
addions mol K+ 0
addions mol Cl- 0

# Write Amber files and PDB for further processing
saveamberparm mol {output_prefix}.prmtop {output_prefix}.inpcrd
savepdb mol {output_prefix}.pdb

quit
"""
    with open("tleap.in", "w") as f:
        f.write(script)
    print("Created tleap.in script.")

def run_tleap():
    """Runs tleap to prepare the system."""
    try:
        result = subprocess.run(["tleap", "-f", "tleap.in"], check=True,
                                capture_output=True, text=True)
        print("tleap output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running tleap:")
        print(e.stdout)
        print(e.stderr)
        raise

def center_and_trim_system(prmtop_file, inpcrd_file, trim_size=15.0):
    """
    Loads the Amber system, centers it, and trims atoms outside a cubic box of size trim_size (nm).
    Returns a new Topology and positions.
    """
    print("\nCentering and trimming system...")
    prmtop = app.AmberPrmtopFile(prmtop_file)
    inpcrd = app.AmberInpcrdFile(inpcrd_file)
    positions = inpcrd.positions

    # Convert positions to a numpy array in nm and center them.
    pos_array = np.array([[x.value_in_unit(unit.nanometer) for x in pos] for pos in positions])
    center = np.mean(pos_array, axis=0)
    centered_pos = pos_array - center

    half_size = trim_size / 2.0
    keep_indices = [i for i, pos in enumerate(centered_pos)
                    if (abs(pos[0]) <= half_size and abs(pos[1]) <= half_size and abs(pos[2]) <= half_size)]

    new_top = app.Topology()
    new_pos = []
    atom_map = {}

    for chain in prmtop.topology.chains():
        new_chain = new_top.addChain(chain.id)
        for residue in chain.residues():
            if any(atom.index in keep_indices for atom in residue.atoms()):
                new_res = new_top.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    if atom.index in keep_indices:
                        new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                        atom_map[atom] = new_atom
                        new_pos.append(unit.Quantity(centered_pos[atom.index], unit.nanometer))
    # Copy bonds among kept atoms.
    for bond in prmtop.topology.bonds():
        if bond.atom1 in atom_map and bond.atom2 in atom_map:
            new_top.addBond(atom_map[bond.atom1], atom_map[bond.atom2])
    print(f"Kept {len(keep_indices)} atoms after trimming.")
    return new_top, new_pos

def prepare_system(pdb_file, ion_type):
    """
    Prepares the system by creating and running the tleap script, centering/triming the system,
    and adding solvent and ions.
    """
    print("Preparing system...")
    create_tleap_script(pdb_file)
    run_tleap()
    topology, positions = center_and_trim_system("system.prmtop", "system.inpcrd")
    modeller = app.Modeller(topology, positions)

    # Calculate system dimensions
    pos_array = np.array([[x.value_in_unit(unit.nanometer) for x in pos] for pos in positions])
    min_pos = np.min(pos_array, axis=0)
    max_pos = np.max(pos_array, axis=0)
    dimensions = max_pos - min_pos
    print(f"System dimensions (nm): {dimensions[0]:.1f} x {dimensions[1]:.1f} x {dimensions[2]:.1f}")

    padding = 0.8  # nm
    box_dimensions = dimensions + 2 * padding
    print(f"Final box size (nm): {box_dimensions[0]:.1f} x {box_dimensions[1]:.1f} x {box_dimensions[2]:.1f}")
    boxSize = Vec3(box_dimensions[0], box_dimensions[1], box_dimensions[2]) * unit.nanometer

    print("Adding solvent and ions...")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    try:
        modeller.addSolvent(forcefield=forcefield,
                            model='tip3p',
                            boxSize=boxSize,
                            ionicStrength=0.15 * unit.molar,
                            positiveIon=ION_PARAMS[ion_type]['resname'],
                            negativeIon='Cl',
                            neutralize=True)
    except Exception as e:
        print("Error during solvation:")
        print(e)
        print(f"Number of atoms: {modeller.topology.getNumAtoms()}")
        raise

    return modeller

# ------------------------------------------------------------------------------
# Simulation and Analysis Functions

def run_simulation(input_pdb, voltage, time_ns, ion_type):
    """
    Runs the MD simulation:
      - Prepares the system.
      - Creates an OpenMM system with PME and constraints.
      - Adds a custom external force to mimic an electric field.
      - Runs a simulation and saves a trajectory.
      - After the simulation, analyzes the trajectory to count ion crossing events.
      - Computes the ionic current and conductance.
    """
    print("Starting simulation...")
    modeller = prepare_system(input_pdb, ion_type)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=app.HBonds)

    # Determine the box length in the z-direction.
    try:
        box_vectors = modeller.topology.getPeriodicBoxVectors()
        box_length_z = box_vectors[2][2]
    except Exception:
        print("Periodic box vectors not found. Using default 10 nm.")
        box_length_z = 10.0 * unit.nanometer

    # Compute electric field (V/nm) such that Voltage = E * box_length_z.
    E_vnm = voltage / box_length_z.value_in_unit(unit.nanometer)
    # Conversion factor: ~96.485 kJ/(mol*nm)/e per V/nm.
    E_param = E_vnm * 96.485  
    print(f"Applying electric field: {E_vnm:.3f} V/nm (E_param = {E_param:.3f} kJ/(mol*nm)/e)")

    # Add a custom external force: U = -charge * E * z.
    custom_force = CustomExternalForce(" - charge * E * z ")
    custom_force.addGlobalParameter("E", E_param)
    custom_force.addPerParticleParameter("charge")

    # Locate the NonbondedForce to extract particle charges.
    nbforce = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nbforce = force
            break
    if nbforce is None:
        raise ValueError("No NonbondedForce found in the system!")

    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nbforce.getParticleParameters(i)
        custom_force.addParticle(i, [charge])
    system.addForce(custom_force)

    # Set up a Langevin integrator (2 fs time step).
    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)

    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    print("Minimizing energy...")
    simulation.minimizeEnergy()

    # Set up reporters to write the trajectory and log data.
    traj_filename = "trajectory.dcd"
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
                                                  potentialEnergy=True, temperature=True))
    simulation.reporters.append(app.DCDReporter(traj_filename, 1000))

    # Determine number of steps. (2 fs per step: 1 ns = 500,000 steps)
    nsteps = int(time_ns * 500000)
    print(f"Running simulation for {nsteps} steps (~{time_ns} ns)...")
    simulation.step(nsteps)

    # Save final state as PDB (for topology in analysis).
    final_pdb = "final_state.pdb"
    with open(final_pdb, "w") as f:
        app.PDBFile.writeFile(simulation.topology,
                              simulation.context.getState(getPositions=True).getPositions(),
                              f)
    print("Simulation complete.")

    # ---------------------
    # Trajectory Analysis:
    n_crossings = analyze_ion_crossings(traj_filename, final_pdb, ion_type, plane_z=0.0)

    # Compute current (I = (# crossings * ion charge) / simulation time).
    # Ion charge (absolute value) in coulombs:
    q_ion = 1.60217662e-19  # Coulomb for a monovalent ion
    sim_time_seconds = time_ns * 1e-9  # convert ns to seconds
    current = (n_crossings * q_ion) / sim_time_seconds  # in Amperes
    conductance = current / voltage  # in Siemens (S)
    conductance_pS = conductance * 1e12  # picosiemens (pS)

    print("\n--- Analysis Results ---")
    print(f"Total simulation time: {time_ns} ns")
    print(f"Number of ion crossing events: {n_crossings}")
    print(f"Ionic current: {current:.3e} A")
    print(f"Conductance: {conductance:.3e} S  ({conductance_pS:.1f} pS)")

    # Log details to a text file.
    with open("conductance_log.txt", "w") as logfile:
        logfile.write(f"Simulation Time (ns): {time_ns}\n")
        logfile.write(f"Applied Voltage (V): {voltage}\n")
        logfile.write(f"Ion type: {ion_type}\n")
        logfile.write(f"Number of crossing events: {n_crossings}\n")
        logfile.write(f"Ionic current (A): {current:.3e}\n")
        logfile.write(f"Conductance (S): {conductance:.3e}  ({conductance_pS:.1f} pS)\n")

    return conductance, n_crossings, current

def analyze_ion_crossings(traj_filename, topology_filename, ion_type, plane_z=0.0):
    """
    Analyzes the trajectory (DCD) with MDAnalysis to count the number of ion crossing events.
    For positive ions (K+, Na+): counts events when an ion goes from below to above the plane (z from < plane_z to >= plane_z).
    For Cl-: counts events when an ion goes from above to below (z from > plane_z to <= plane_z).
    """
    print("Analyzing trajectory for ion crossing events...")
    # Load the Universe (using the final PDB as topology)
    u = mda.Universe(topology_filename, traj_filename)
    if ion_type == "K+":
        ion_selection = "resname K"
    elif ion_type == "Na+":
        ion_selection = "resname Na"
    elif ion_type == "Cl-":
        ion_selection = "resname Cl"
    else:
        raise ValueError("Invalid ion type specified.")
    ions = u.select_atoms(ion_selection)
    print(f"Found {len(ions)} {ion_type} ions for analysis.")

    # Prepare a dictionary to store each ion's z positions over the trajectory.
    ion_indices = ions.indices
    ion_z = {idx: [] for idx in ion_indices}
    # Iterate over trajectory frames.
    for ts in u.trajectory:
        for idx in ion_indices:
            pos_z = u.atoms[idx].position[2]  # position in nm (OpenMM PDBFile writes nm)
            ion_z[idx].append(pos_z)

    # Count crossings for each ion.
    total_crossings = 0
    for idx, z_list in ion_z.items():
        z_array = np.array(z_list)
        crossings = 0
        if ion_type in ["K+", "Na+"]:
            # Count when z goes from below plane_z to above or equal.
            for i in range(len(z_array) - 1):
                if (z_array[i] < plane_z) and (z_array[i + 1] >= plane_z):
                    crossings += 1
        elif ion_type == "Cl-":
            # For negative ions, count when z goes from above plane_z to below or equal.
            for i in range(len(z_array) - 1):
                if (z_array[i] > plane_z) and (z_array[i + 1] <= plane_z):
                    crossings += 1
        total_crossings += crossings
    print(f"Total crossing events counted: {total_crossings}")
    return total_crossings

# ------------------------------------------------------------------------------
# Main execution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ion Channel Conductance Calculator")
    parser.add_argument("input_pdb", help="Membrane-embedded PDB file")
    parser.add_argument("--voltage", type=float, required=True, help="Voltage (V)")
    parser.add_argument("--time_ns", type=float, default=20.0, help="Simulation time (ns)")
    parser.add_argument("--ion", choices=["K+", "Na+", "Cl-"], required=True, help="Ion type")
    args = parser.parse_args()

    try:
        run_simulation(args.input_pdb, args.voltage, args.time_ns, args.ion)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
