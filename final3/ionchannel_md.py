#!/usr/bin/env python3
"""
fix_embed_run_conductance.py

Usage:
  python fix_embed_run_conductance.py raw_channel.pdb

What It Does:
  1) Fixes a raw/partial PDB with PDBFixer (missing atoms, residues, hydrogens).
  2) Embeds the protein/ion channel in a POPC membrane (Z-axis normal).
  3) Solvates the system (TIP3P water).
  4) Runs a short OpenMM simulation (minimize + ~20 ps of MD).
  5) After the simulation, uses a naive geometry-based method to estimate
     the single-channel conductance for each of several ions (K+, Na+, Cl-, Ca2+).

Outputs:
  - channel_membrane.pdb  : The built system (channel + membrane + water)
  - traj.dcd              : Short MD trajectory
  - log.txt               : Temperature/Energy log
  - final_positions.pdb   : Final snapshot after MD
  - Conductance estimates printed to stdout
"""

import sys
import os

# --- PDBFixer for building/fixing ---
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# --- OpenMM imports ---
import openmm
import openmm.app as app
import openmm.unit as unit

# --- For geometric analysis (pore radius, length) ---
import math
import numpy as np

# ------------------------------------------------------
# 1) ION MOBILITIES (rough bulk values)
#    For a geometry-based conduction estimate
# ------------------------------------------------------
ION_MOBILITIES = {
    "K+":   7.62e-8,  # m^2/(s·V)
    "Na+":  5.19e-8,
    "Ca2+": 6.17e-8,  # ignoring the fact it's divalent, etc.
    "Cl-":  7.91e-8
}

# We'll assume integer charges:
ION_CHARGES = {
    "K+":   1,
    "Na+":  1,
    "Cl-": -1,
    "Ca2+": 2
}

FARADAY_CONST = 96485  # C/mol

def approximate_conductance(radius_m, length_m, ion="K+"):
    """
    Very naive conductance estimate using:
       G ~ z * F * mu * (pi * r^2 / length)
    in units of Siemens (S).
    
    radius_m: pore radius (meters)
    length_m: pore length (meters)
    ion:      string in ["K+", "Na+", "Ca2+", "Cl-"]
    """
    if ion not in ION_MOBILITIES or ion not in ION_CHARGES:
        raise ValueError(f"Ion '{ion}' not recognized.")
    mu = ION_MOBILITIES[ion]         # m^2/(s·V)
    z  = ION_CHARGES[ion]            # unitless, e.g. +1, +2, -1
    area = math.pi * radius_m**2
    # G = |z| * F * mu * (area / length)
    G = abs(z)*FARADAY_CONST*mu*(area/length_m)
    return G

def compute_pore_geometry(pdb_file, zmin=-5.0, zmax=5.0):
    """
    Reads a PDB, looks for alpha-carbons (CA) whose z-coordinates are
    between zmin and zmax (in Angstroms), then estimates:
      - pore length = maxZ - minZ
      - pore radius = (avg distance from center in x-y) + 1 std dev
    Returns (radius_m, length_m) in meters.
    
    NOTE: This is naive. Tools like 'HOLE' or 'MDAnalysis' are better for real analysis.
    """
    pdb = app.PDBFile(pdb_file)
    coords = []
    for atom, res in zip(pdb.topology.atoms(), pdb.positions):
        if atom.name == 'CA':
            # res is an openmm.Vec3 with units=nanometers
            # Convert to Angstrom: 1 nm = 10 Å
            x_angs = res.x*10.0
            y_angs = res.y*10.0
            z_angs = res.z*10.0
            if zmin < z_angs < zmax:
                coords.append((x_angs, y_angs, z_angs))
    if not coords:
        raise ValueError("No alpha carbons found in the specified z-range.")

    coords = np.array(coords)
    z_vals = coords[:,2]
    length_A = z_vals.max() - z_vals.min()

    # find centroid in x-y
    x_mean = np.mean(coords[:,0])
    y_mean = np.mean(coords[:,1])

    # distance from centroid in x-y plane
    r_array = []
    for (x,y,z) in coords:
        rr = math.sqrt((x - x_mean)**2 + (y - y_mean)**2)
        r_array.append(rr)
    r_mean = np.mean(r_array)
    r_std  = np.std(r_array)
    pore_radius_A = r_mean + r_std

    # convert Angstrom->meter
    # 1 Angstrom = 1e-10 m
    pore_radius_m = pore_radius_A * 1e-10
    pore_length_m = length_A    * 1e-10
    return pore_radius_m, pore_length_m


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_pdb>")
        sys.exit(1)

    input_pdb = sys.argv[1]
    if not os.path.isfile(input_pdb):
        print(f"Error: file '{input_pdb}' does not exist.")
        sys.exit(1)

    # ---------------------------------------------------------
    # PART A: FIX AND EMBED IN MEMBRANE VIA PDBFixer
    # ---------------------------------------------------------
    print("Loading and fixing PDB using PDBFixer...")
    fixer = PDBFixer(filename=input_pdb)

    # Identify missing residues/atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    # Add missing heavy atoms + hydrogens
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    # Embed protein in a POPC membrane (Z-axis normal)
    print("Embedding in a POPC membrane...")
    fixer.addMembrane(forcefield='charmm36.xml', lipidType='POPC')
    
    # Solvate (add water) on both sides of the membrane
    print("Adding water (TIP3P)...")
    fixer.addSolvent(forcefield='charmm36.xml', model='tip3p', padding=1.0)

    # Write out the new PDB with protein + membrane + water
    output_pdb = 'channel_membrane.pdb'
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"System with membrane written to '{output_pdb}'")

    # ---------------------------------------------------------
    # PART B: SET UP OPENMM SIMULATION
    # ---------------------------------------------------------
    print("Setting up OpenMM simulation...")

    pdb = app.PDBFile(output_pdb)
    forcefield = app.ForceField('charmm36.xml')

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=app.HBonds
    )

    # Add a barostat for constant pressure (NPT)
    system.addForce(openmm.MonteCarloBarostat(1.0*unit.bar, 310*unit.kelvin))

    integrator = openmm.LangevinIntegrator(
        310*unit.kelvin, 
        1.0/unit.picosecond,
        0.002*unit.picoseconds
    )

    # GPU platform
    platform = openmm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)

    print("Minimizing energy...")
    simulation.minimizeEnergy()

    # Add reporters
    simulation.reporters.append(app.StateDataReporter(
        'log.txt', 
        1000, 
        step=True,
        potentialEnergy=True,
        temperature=True,
        density=True
    ))
    simulation.reporters.append(app.DCDReporter('traj.dcd', 1000))

    print("Running MD for 10,000 steps (~20 ps)...")
    simulation.step(10000)

    # Write final positions
    final_pos = simulation.context.getState(getPositions=True).getPositions()
    final_pdb = 'final_positions.pdb'
    with open(final_pdb, 'w') as f:
        app.PDBFile.writeFile(simulation.topology, final_pos, f)

    print("MD run complete. Final snapshot in 'final_positions.pdb'")

    # ---------------------------------------------------------
    # PART C: NAIVE GEOMETRY-BASED CONDUCTANCE ESTIMATE
    # ---------------------------------------------------------
    print("\n--- Naive Conductance Estimation ---")

    # We'll parse the final PDB file (which includes the entire system),
    # but we only look for alpha-carbons in a small Z-range around the pore
    # (zmin=-5, zmax=5) to guess the channel region.
    # Adjust these if your channel is longer or differently oriented.
    try:
        radius_m, length_m = compute_pore_geometry(final_pdb, zmin=-5, zmax=5)
        print(f"Pore radius ~ {radius_m*1e10:.2f} Å, length ~ {length_m*1e10:.2f} Å")
    except ValueError as e:
        print(f"Error computing pore geometry: {e}")
        print("Cannot compute conductance.")
        return

    # For each ion of interest, compute approximate G
    ions_of_interest = ["K+", "Na+", "Cl-", "Ca2+"]
    for ion in ions_of_interest:
        try:
            G = approximate_conductance(radius_m, length_m, ion=ion)
            # G in Siemens
            print(f"  {ion:4s} conductance ~ {G:.3e} S")
        except ValueError as err:
            print(f"Could not compute conduction for {ion}: {err}")

    print("\nDone!")

if __name__ == "__main__":
    main()
