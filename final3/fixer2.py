#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import logging
import warnings

from Bio.PDB import PDBParser, Structure, Model, Chain
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter("ignore", PDBConstructionWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# 1) GLOBAL CONSTANTS & DUMMY PROPERTIES
# ----------------------------------------------------------------

# Faraday's constant
FARADAY = 96485.0  # C/mol

# RT at ~310 K in kcal/mol (roughly 37°C)
RT_310K = 0.5924

# Example Ion Properties
# These are GENERIC placeholders. You can modify them if you learn
# better values for your channel of interest.
ION_PROPERTIES = {
    'K': {
        'charge':  1,
        'radius':  1.33,  # ionic radius (Å)
        'min_radius': 1.3,
        'mobility': 7.62e-8,       # m^2/(V·s)
        'selectivity': 1.0,        # relative to other ions
        'dehydration_energy': 2.0  # kcal/mol
    },
    'Na': {
        'charge':  1,
        'radius':  0.95,
        'min_radius': 1.0,
        'mobility': 5.19e-8,
        'selectivity': 0.05,
        'dehydration_energy': 3.0
    },
    'Ca': {
        'charge':  2,
        'radius':  0.99,
        'min_radius': 1.1,
        'mobility': 6.17e-8,
        'selectivity': 0.01,
        'dehydration_energy': 4.0
    },
    'Cl': {
        'charge': -1,
        'radius': 1.81,
        'min_radius': 1.8,
        'mobility': 7.91e-8,
        'selectivity': 0.001,
        'dehydration_energy': 3.5
    }
}

# Typical concentrations (mM) -> 1 mM = 1 mol/m^3
CONCENTRATIONS_MM = {
    'K':  145.0,
    'Na': 12.0,
    'Ca': 1.0,    # Just a placeholder
    'Cl': 4.0
}

# Empirical "fudge" factors
# These amplify or reduce conduction to approximate known channel behavior.
EMPIRICAL_FACTORS = {
    'K':  12.0,
    'Na': 0.1,
    'Ca': 0.01,
    'Cl': 0.001
}

# Simple table of van der Waals radii by element (Å), for collision checks
# If your channel has exotic atoms, add them here with approximate radii.
VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "K": 2.75, "NA": 2.27, "MG": 1.73, "CA": 1.94,
    "CL": 1.75, "ZN": 1.39
}


# ----------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ----------------------------------------------------------------

def parse_axis(axis_arg):
    """
    Interpret the --axis argument:
      - "auto" => return None (means we do PCA)
      - "x,y,z" => parse as a float 3-vector
    """
    if axis_arg.lower() == "auto":
        return None
    vals = axis_arg.split(",")
    if len(vals) != 3:
        raise ValueError("Axis must be 'auto' or three comma-separated floats, e.g. 0,0,1")
    arr = np.array([float(v) for v in vals])
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        raise ValueError("Supplied axis vector has near-zero length.")
    return arr / norm


def filter_structure_by_chains(structure, chain_ids=None):
    """
    Return a new structure containing ONLY the specified chain IDs
    (if chain_ids is not None). If chain_ids is None, we keep all chains.
    """
    if not chain_ids:
        return structure  # keep everything

    new_struct = Structure.Structure("filtered")
    new_model = Model.Model(0)
    new_struct.add(new_model)

    for chain in structure.get_chains():
        if chain.id in chain_ids:
            copy_chain = Chain.Chain(chain.id)
            for residue in chain:
                copy_chain.add(residue)
            new_model.add(copy_chain)

    return new_struct


# ----------------------------------------------------------------
# 3) MAIN ANALYZER CLASS
# ----------------------------------------------------------------

class IonChannelAnalyzer:
    """
    A generic class to:
      1) Load/parse PDB
      2) Possibly filter out chains not of interest
      3) Perform PCA or set user axis
      4) Compute pore radius
      5) Estimate single-channel conductance
    """

    def __init__(self, pdb_path, chain_ids=None, axis=None, channel_length=40.0):
        """
        :param pdb_path: path to PDB file
        :param chain_ids: list or set of chain IDs to keep (None => keep all)
        :param axis: None => PCA, or a 3-vector => conduction axis
        :param channel_length: length of channel in Å (default 40)
        """
        self.pdb_path = pdb_path
        self.chain_ids = chain_ids
        self.channel_length = channel_length

        parser = PDBParser(QUIET=True)
        full_structure = parser.get_structure("channel", pdb_path)

        # Keep only selected chains if specified
        self.structure = filter_structure_by_chains(full_structure, chain_ids)

        self._print_structure_info()
        self._setup_axis(axis)
        self._setup_atom_data()

    def _print_structure_info(self):
        atoms = list(self.structure.get_atoms())
        residues = list(self.structure.get_residues())
        chains = list(self.structure.get_chains())
        logger.info(f"Structure loaded from: {self.pdb_path}")
        logger.info(f"Chains: {len(chains)}  |  Residues: {len(residues)}  |  Atoms: {len(atoms)}")

    def _setup_axis(self, user_axis):
        """
        If user_axis is None => do PCA
        Otherwise, set channel_axis to user_axis
        Also compute channel_center as the mean of all atom coords
        """
        # Collect all coords
        coords = []
        for atom in self.structure.get_atoms():
            coords.append(atom.get_coord())
        coords = np.array(coords)

        if coords.size == 0:
            raise ValueError("No atoms found; check your PDB or chain IDs.")

        # Center of mass (simple average)
        self.channel_center = coords.mean(axis=0)

        if user_axis is None:
            # Perform PCA to find largest principal axis
            coords_centered = coords - self.channel_center
            cov = np.cov(coords_centered.T)
            e_vals, e_vecs = np.linalg.eig(cov)
            idx_max = np.argmax(e_vals)
            principal_axis = e_vecs[:, idx_max]
            principal_axis /= np.linalg.norm(principal_axis)
            self.channel_axis = principal_axis
            logger.info(f"Channel axis (PCA): {self.channel_axis}")
            logger.info(f"Channel center (mean CoM): {self.channel_center}")
        else:
            # user-supplied axis
            self.channel_axis = user_axis
            logger.info(f"User axis: {self.channel_axis}")
            logger.info(f"Channel center (mean position): {self.channel_center}")

    def _setup_atom_data(self):
        """
        For each atom, record:
          - its z-projection along channel_axis
          - its XY vector (relative to channel_axis)
          - approximate van der Waals radius
        """
        self.atom_data = []
        for atom in self.structure.get_atoms():
            coord = atom.get_coord()
            element = atom.element.strip().upper()
            # fallback to 1.70 if unknown
            vdw_r = VDW_RADII.get(element, 1.70)

            rel = coord - self.channel_center
            z_val = np.dot(rel, self.channel_axis)
            xy_vec = rel - z_val * self.channel_axis

            self.atom_data.append({
                'z': z_val,
                'xy': xy_vec,
                'radius': vdw_r
            })

        logger.info(f"Preprocessed {len(self.atom_data)} atoms in channel coordinates.")

    def _get_pore_radius(self, z_val, z_tolerance=2.0, search_radius=10.0):
        """
        Return the maximum radius in the XY plane at the given z_val
        that doesn't collide with atoms whose z is within ± z_tolerance.
        Does a coarse 2D grid search, then refines.
        """
        near_atoms = [a for a in self.atom_data if abs(a['z'] - z_val) < z_tolerance]
        if not near_atoms:
            return 0.0

        best_r = 0.0
        steps = 10
        grid_lin = np.linspace(-search_radius, search_radius, steps)

        for gx in grid_lin:
            for gy in grid_lin:
                pt = np.array([gx, gy])
                try:
                    candidate = min(np.linalg.norm(a['xy'][:2] - pt) - a['radius'] for a in near_atoms)
                except ValueError:
                    candidate = 0.0
                if candidate > best_r:
                    # refine locally
                    finer = 5
                    for fx in np.linspace(-1, 1, finer):
                        for fy in np.linspace(-1, 1, finer):
                            rpt = pt + np.array([fx, fy])
                            col = min(np.linalg.norm(a['xy'][:2] - rpt) - a['radius']
                                      for a in near_atoms)
                            if col > best_r:
                                best_r = col

        return max(best_r, 0.0)

    def estimate_conductance(self, ion):
        """
        Estimate channel conductance for one ion (K, Na, Ca, Cl, etc.).
        Returns conductance in Siemens (S).
        """
        if ion not in ION_PROPERTIES:
            raise ValueError(f"No property data for ion: {ion}")
        props = ION_PROPERTIES[ion]

        # Scan z in [-20..20], find minimum pore radius
        z_points = np.linspace(-20, 20, 21)
        radii = [self._get_pore_radius(z) for z in z_points]
        min_r = min(radii) if radii else 0.0
        logger.info(f"[{ion}] Bottleneck radius ~ {min_r:.2f} Å")

        # Size exclusion
        if min_r < props['min_radius']:
            logger.info(f"Ion {ion} fails size constraint (needs {props['min_radius']:.2f} Å).")
            return 0.0

        # Dehydration factor
        E_dehyd = props['dehydration_energy']
        dehyd_factor = np.exp(-E_dehyd / RT_310K)

        # Convert geometry to meters
        length_m = (self.channel_length * 1e-10)  # Å -> m
        area_m2  = np.pi * (min_r * 1e-10)**2

        # Ion mobility, valence, selectivity
        z_val   = abs(props['charge'])
        mu      = props['mobility']
        sel     = props['selectivity']

        # Bulk concentration
        c_m3 = CONCENTRATIONS_MM.get(ion, 1.0)  # fallback to 1.0 mM if unknown

        # Base conductance in S
        G_base = z_val * FARADAY * mu * c_m3 * (area_m2 / length_m) * sel

        # Apply dehydration
        G_dehyd = G_base * dehyd_factor

        # Apply empirical factor
        emp_factor = EMPIRICAL_FACTORS.get(ion, 1.0)
        G_final = G_dehyd * emp_factor

        logger.info(f"[{ion}] Base G: {G_base*1e12:.3f} pS | "
                    f"After dehydration: {G_dehyd*1e12:.3f} pS | "
                    f"Final: {G_final*1e12:.3f} pS")
        return G_final

    def analyze_ions(self, ion_list):
        """
        Analyze multiple ions and return a Pandas DataFrame.
        """
        results = {}
        for ion in ion_list:
            g_siemens = self.estimate_conductance(ion)
            results[ion] = f"{g_siemens * 1e12:.2f} pS"
        return pd.DataFrame.from_dict(results, orient='index', columns=['Conductance (pS)'])


# ----------------------------------------------------------------
# 4) CLI MAIN FUNCTION
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute approximate single-channel conductance from a PDB file "
                    "(with no experimental data required)."
    )
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--chains", default="", 
                        help="Comma-separated chain IDs to keep, e.g. A,B. Blank => keep all.")
    parser.add_argument("--axis", default="auto",
                        help="'auto' => PCA axis, or 'x,y,z' => manually specify. Example: 0,0,1")
    parser.add_argument("--ions", default="K,Na,Ca,Cl",
                        help="Comma-separated ions to analyze, e.g. K,Na,Ca,Cl")
    parser.add_argument("--length", default=40.0, type=float,
                        help="Channel conduction path length in Å (default 40)")

    args = parser.parse_args()

    # Parse chain IDs
    chain_ids = None
    if args.chains.strip():
        chain_ids = [c.strip() for c in args.chains.split(",")]

    # Parse axis
    user_axis = parse_axis(args.axis)

    # Parse ions
    ions = [i.strip() for i in args.ions.split(",")]

    # Run analysis
    analyzer = IonChannelAnalyzer(
        pdb_path=args.pdb,
        chain_ids=chain_ids,
        axis=user_axis,
        channel_length=args.length
    )

    df = analyzer.analyze_ions(ions)

    # Print results
    print("\n=== Single-Channel Conductance Estimates ===")
    print(df)
    print("\nDisclaimer: All parameters are placeholders and not based on "
          "actual experimental data for your channel. Adjust ION_PROPERTIES "
          "and EMPIRICAL_FACTORS to better reflect reality.")


if __name__ == "__main__":
    main()


# python fixer2.py --pdb 7zdznew.pdb --chains A   --ions K,Na,Ca,Cl  --axis auto  --length 40
# python fixer2.py --pdb 6j8e.pdb --chains A   --ions K,Na,Ca,Cl  --axis auto  --length 40