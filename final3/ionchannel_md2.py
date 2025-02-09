#!/usr/bin/env python3
"""
Neuro Ion Channel Conductance Calculator (PDBFixer/OpenMM Compatible)

Usage:
  python neuro_conductance.py input.pdb --voltage 0.065 --time_ns 20 --ion K+
"""

import sys
import argparse
import numpy as np
from pdbfixer import PDBFixer
from openmm import app, unit, Platform, LangevinIntegrator, Vec3
from openmm.app import PDBFile, Modeller, ForceField, PME, HBonds, DCDReporter, PDBReporter
import mdtraj as md

# Neuroscience ion parameters (Amber)
ION_PARAMS = {
    'K+': {
        'resname': 'K',
        'charge': 1.0 * unit.elementary_charge,
        'sigma': 0.303763 * unit.nanometer,  # Amber parameters
        'epsilon': 0.36923 * unit.kilojoule_per_mole,
    },
    'Na+': {
        'resname': 'NA',
        'charge': 1.0 * unit.elementary_charge,
        'sigma': 0.212557 * unit.nanometer,
        'epsilon': 0.0874393 * unit.kilojoule_per_mole,
    },
    'Ca2+': {
        'resname': 'CA',
        'charge': 2.0 * unit.elementary_charge,
        'sigma': 0.241833 * unit.nanometer,
        'epsilon': 0.50208 * unit.kilojoule_per_mole,
    },
    'Cl-': {
        'resname': 'CL',
        'charge': -1.0 * unit.elementary_charge,
        'sigma': 0.440104 * unit.nanometer,
        'epsilon': 0.41840 * unit.kilojoule_per_mole,
    }
}

def remove_non_standard_residues(modeller):
    """Remove non-standard residues using Modeller"""
    to_delete = []
    for res in modeller.topology.residues():
        if res.name in ['SR', 'K']:  # Add other non-standard residues as needed
            to_delete.append(res)
    
    if to_delete:
        print(f"Removing {len(to_delete)} non-standard residues...")
        modeller.delete(to_delete)
    return modeller

def prepare_system(pdb_file, ion_type):
    """Prepare membrane-embedded system with neuro ions"""
    print("Loading PDB file...")
    fixer = PDBFixer(filename=pdb_file)
    
    print("Finding missing residues...")
    fixer.findMissingResidues()
    
    print("Finding missing atoms...")
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    
    print("Adding missing atoms...")
    fixer.addMissingAtoms()
    
    print("Adding missing hydrogens...")
    fixer.addMissingHydrogens(7.0)
    
    # Create modeller and remove non-standard residues
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller = remove_non_standard_residues(modeller)
    
    # Save intermediate structure
    print("Saving processed structure...")
    PDBFile.writeFile(modeller.topology, modeller.positions, open('processed.pdb', 'w'))
    
    print("Adding solvent and ions...")
    # Modern solvent/ion addition
    modeller.addSolvent(
        forcefield=ForceField('amber14-all.xml', 'amber14/tip3p.xml'),
        model='tip3p',
        padding=1.2*unit.nanometer,
        ionicStrength=0.15*unit.molar,
        positiveIon=ION_PARAMS[ion_type]['resname'],
        negativeIon='CL',  # Changed from CLA to CL for Amber
        neutralize=True
    )
    
    print("System preparation complete.")
    return modeller

def configure_forcefield():
    """Create force field with neuro ion parameters"""
    # Standard OpenMM force fields
    ff = ForceField(
        'amber14-all.xml',
        'amber14/tip3p.xml',
        'amber14/tip3p_standard.xml'
    )
    
    # Validate available residues
    known_residues = set()
    for template in ff._templates.values():
        known_residues.add(template.name)
    print(f"\nForce field includes {len(known_residues)} residue templates")
    return ff

def calculate_conductance(traj_file, topology_file, voltage, ion_type, time_ns):
    """Calculate conductance from ion flux"""
    print("Analyzing trajectory...")
    ion = ION_PARAMS[ion_type]
    traj = md.load(traj_file, top=topology_file)
    
    # Find ion residues
    ion_indices = [a.index for a in traj.topology.atoms 
                  if a.residue.name == ion['resname']]
    
    if not ion_indices:
        print(f"Warning: No {ion['resname']} ions found in trajectory")
        return 0 * unit.siemens
    
    # Calculate membrane crossing events
    crossings = 0
    z_prev = traj.xyz[0, ion_indices, 2].mean()
    
    for frame in traj.xyz[1:]:
        z_current = frame[ion_indices, 2].mean()
        if (z_current > 0 and z_prev < 0) or (z_current < 0 and z_prev > 0):
            crossings += 1
        z_prev = z_current
    
    # Conductance calculation
    dt = time_ns * unit.nanoseconds
    current = (crossings * unit.elementary_charge) / dt
    conductance = current / (voltage * unit.volt)
    
    return conductance

def run_simulation(input_pdb, voltage, time_ns, ion_type):
    if ion_type not in ION_PARAMS:
        raise ValueError(f"Supported ions: {', '.join(ION_PARAMS.keys())}")
    
    # 1. System preparation
    print("Preparing system...")
    modeller = prepare_system(input_pdb, ion_type)
    
    # 2. Force field setup
    ff = configure_forcefield()
    
    # 3. Create simulation system
    print("Creating simulation system...")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        removeCMMotion=True
    )
    
    # 4. Add voltage potential
    if voltage != 0:
        print(f"Adding voltage bias: {voltage} V")
        box_z = modeller.topology.getUnitCellDimensions()[2].value_in_unit(unit.nanometer)
        electric_field = (voltage * unit.volt) / (box_z * unit.nanometer)
        
        from openmm import CustomExternalForce
        ef_force = CustomExternalForce("q*E*z")
        ef_force.addGlobalParameter("E", electric_field)
        ef_force.addPerParticleParameter("q")
        
        nb_force = None
        for force in system.getForces():
            if isinstance(force, app.NonbondedForce):
                nb_force = force
                break
        
        if nb_force:
            for i in range(system.getNumParticles()):
                charge, _, _ = nb_force.getParticleParameters(i)
                ef_force.addParticle(i, [charge])
            system.addForce(ef_force)
        else:
            print("Warning: Could not find NonbondedForce")

    # 5. Simulation setup
    print("Setting up simulation...")
    integrator = LangevinIntegrator(
        310*unit.kelvin,
        1/unit.picosecond,
        0.002*unit.picoseconds
    )
    
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}
    
    simulation = app.Simulation(
        modeller.topology,
        system,
        integrator,
        platform,
        properties
    )
    
    simulation.context.setPositions(modeller.positions)
    
    # 6. Minimization
    print("Energy minimization...")
    simulation.minimizeEnergy(maxIterations=1000)
    
    # Save minimized structure
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open('minimized.pdb', 'w'))
    
    # 7. Production MD
    print(f"Running {time_ns} ns simulation...")
    nsteps = int(time_ns * 500000)  # 2 fs timestep
    
    simulation.reporters.append(DCDReporter('trajectory.dcd', 1000))
    simulation.reporters.append(app.StateDataReporter(
        'output.log', 1000, step=True, time=True, 
        potentialEnergy=True, temperature=True, speed=True
    ))
    
    simulation.step(nsteps)
    
    # 8. Analysis
    print("Calculating conductance...")
    conductance = calculate_conductance(
        'trajectory.dcd',
        'minimized.pdb',
        voltage,
        ion_type,
        time_ns
    )
    
    print(f"\n{ion_type} Conductance: {conductance.value_in_unit(unit.picosiemens):.2f} pS")
    return conductance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ion Channel Conductance Calculator")
    parser.add_argument("input_pdb", help="Membrane-embedded PDB file")
    parser.add_argument("--voltage", type=float, required=True, help="Voltage (V)")
    parser.add_argument("--time_ns", type=float, default=20.0, help="Simulation time (ns)")
    parser.add_argument("--ion", choices=list(ION_PARAMS.keys()), required=True, help="Ion type")
    
    args = parser.parse_args()
    
    try:
        run_simulation(args.input_pdb, args.voltage, args.time_ns, args.ion)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)