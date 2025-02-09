import numpy as np
from Bio.PDB import *
import pandas as pd
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import logging
from typing import Dict, List, Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress BioPython warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

class IonChannelProperties:
    """Database of ion channel properties"""
    
    # Channel family characteristics
    CHANNEL_FAMILIES = {
        'K': {
            'pattern': r'K(v|ir|Ca|2P|Na)?\d*\.\d*',
            'primary_ion': 'K',
            'base_conductance': 35e-12,  # ~35 pS for K+ channels
            'selectivity_sequence': ['K', 'Rb', 'Na', 'Li', 'Cs'],
            'pore_characteristics': {
                'typical_radius': 1.5,    # Å
                'selectivity_filter': True,
                'filter_sequence': ['TVGYG']  # Common K+ channel filter
            }
        },
        'TRP': {
            'pattern': r'TRP[CVMA]\d*',
            'primary_ion': 'Ca',
            'base_conductance': 50e-12,  # ~50 pS for TRP channels
            'selectivity_sequence': ['Ca', 'Na', 'K', 'Mg'],
            'pore_characteristics': {
                'typical_radius': 3.0,    # Å
                'selectivity_filter': False
            }
        },
        'Ca': {
            'pattern': r'Ca[v]\d*\.\d*',
            'primary_ion': 'Ca',
            'base_conductance': 20e-12,  # ~20 pS for Ca2+ channels
            'selectivity_sequence': ['Ca', 'Ba', 'Sr', 'Mn'],
            'pore_characteristics': {
                'typical_radius': 2.0,    # Å
                'selectivity_filter': True,
                'filter_sequence': ['EEEE']  # Common Ca2+ channel filter
            }
        },
        'Cl': {
            'pattern': r'(CaCC|ClC)-?\d*',
            'primary_ion': 'Cl',
            'base_conductance': 10e-12,  # ~10 pS for Cl- channels
            'selectivity_sequence': ['Cl', 'Br', 'I', 'F'],
            'pore_characteristics': {
                'typical_radius': 2.5,    # Å
                'selectivity_filter': False
            }
        },
        'P2X': {
            'pattern': r'P2X\d*',
            'primary_ion': 'Ca',
            'base_conductance': 15e-12,
            'selectivity_sequence': ['Ca', 'Na', 'K'],
            'pore_characteristics': {
                'typical_radius': 2.8,    # Å
                'selectivity_filter': False
            }
        },
        'Piezo': {
            'pattern': r'Piezo\d*',
            'primary_ion': 'Ca',
            'base_conductance': 25e-12,
            'selectivity_sequence': ['Ca', 'Na', 'K'],
            'pore_characteristics': {
                'typical_radius': 3.0,    # Å
                'selectivity_filter': False
            }
        },
        'IP3R': {
            'pattern': r'IP3R\d*',
            'primary_ion': 'Ca',
            'base_conductance': 100e-12,
            'selectivity_sequence': ['Ca', 'K', 'Na'],
            'pore_characteristics': {
                'typical_radius': 3.5,    # Å
                'selectivity_filter': True
            }
        }
    }
    
    # Ion properties
    ION_PROPERTIES = {
        'K': {
            'radius': 1.33,
            'hydrated_radius': 2.32,
            'min_radius': 1.0,
            'charge': 1,
            'mobility': 7.62e-8,
            'dehydration_energy': 2.0
        },
        'Na': {
            'radius': 0.95,
            'hydrated_radius': 2.76,
            'min_radius': 0.9,
            'charge': 1,
            'mobility': 5.19e-8,
            'dehydration_energy': 3.0
        },
        'Ca': {
            'radius': 0.99,
            'hydrated_radius': 4.12,
            'min_radius': 1.0,
            'charge': 2,
            'mobility': 6.17e-8,
            'dehydration_energy': 4.0
        },
        'Cl': {
            'radius': 1.81,
            'hydrated_radius': 3.32,
            'min_radius': 1.2,
            'charge': -1,
            'mobility': 7.91e-8,
            'dehydration_energy': 2.5,
            'relative_permeability': 1.0
        },
        'Br': {
            'radius': 1.96,
            'hydrated_radius': 3.30,
            'min_radius': 1.3,
            'charge': -1,
            'mobility': 8.09e-8,
            'dehydration_energy': 2.3,
            'relative_permeability': 1.1
        },
        'I': {
            'radius': 2.20,
            'hydrated_radius': 3.31,
            'min_radius': 1.4,
            'charge': -1,
            'mobility': 7.96e-8,
            'dehydration_energy': 2.1,
            'relative_permeability': 1.5
        },
        'F': {
            'radius': 1.33,
            'hydrated_radius': 3.52,
            'min_radius': 1.1,
            'charge': -1,
            'mobility': 5.70e-8,
            'dehydration_energy': 2.7,
            'relative_permeability': 0.4
        },
        'Mg': {
            'radius': 0.65,
            'hydrated_radius': 4.28,
            'min_radius': 0.9,
            'charge': 2,
            'mobility': 5.50e-8,
            'dehydration_energy': 4.5
        }
    }
    
    @classmethod
    def identify_channel_family(cls, channel_name: str) -> str:
        """Identify channel family from name"""
        for family, props in cls.CHANNEL_FAMILIES.items():
            if re.match(props['pattern'], channel_name):
                return family
        return 'Unknown'
    
    @classmethod
    def get_selectivity_sequence(cls, family: str) -> List[str]:
        """Get selectivity sequence for channel family"""
        return cls.CHANNEL_FAMILIES.get(family, {}).get('selectivity_sequence', [])
    
    @classmethod
    def get_ion_properties(cls, ion: str) -> Dict:
        """Get properties for specific ion"""
        return cls.ION_PROPERTIES.get(ion, {})

class UniversalChannelAnalyzer:
    def __init__(self, pdb_file: str, channel_name: str):
        """Initialize analyzer with PDB file and channel name"""
        self.pdb_file = pdb_file
        self.channel_name = channel_name
        self.channel_family = IonChannelProperties.identify_channel_family(channel_name)
        self.family_props = IonChannelProperties.CHANNEL_FAMILIES.get(
            self.channel_family, IonChannelProperties.CHANNEL_FAMILIES['K'])
        
        # Initialize structure
        self.parser = PDBParser()
        self.structure = self.parser.get_structure('channel', pdb_file)
        
        # Setup analysis
        self.print_structure_info()
        self.calculate_channel_axis()
        self.setup_atom_grid()
        self.identify_pore_region()
        
    def print_structure_info(self):
        """Print structure information"""
        self.atom_count = sum(1 for atom in self.structure.get_atoms())
        self.residue_count = sum(1 for residue in self.structure.get_residues())
        self.chain_count = len(list(self.structure.get_chains()))
        
        logger.info(f"Analyzing {self.channel_name} ({self.channel_family} family)")
        logger.info(f"Structure contains:")
        logger.info(f"Chains: {self.chain_count}")
        logger.info(f"Residues: {self.residue_count}")
        logger.info(f"Atoms: {self.atom_count}")
        
    def calculate_channel_axis(self):
        """Calculate channel axis using symmetry and backbone atoms"""
        coords = []
        weights = []
        
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                        weights.append(1.0)
        
        coords = np.array(coords)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        mean_coord = np.average(coords, weights=weights, axis=0)
        coords_centered = coords - mean_coord
        cov_matrix = np.cov(coords_centered.T, aweights=weights)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        self.channel_axis = eigenvecs[:, np.argmax(eigenvals)]
        self.channel_center = mean_coord
        
    def setup_atom_grid(self):
        """Setup optimized atom grid for pore analysis"""
        self.atom_data = []
        
        for atom in self.structure.get_atoms():
            if atom.is_disordered():
                continue
                
            coord = atom.get_coord()
            rel_pos = coord - self.channel_center
            
            # Use detailed atom typing
            element = atom.element.upper() if hasattr(atom, 'element') else atom.name[0].upper()
            if element in ['C', 'N', 'O', 'S']:
                radius = 1.4 if atom.get_name() in ['CA', 'C', 'N', 'O'] else 1.6
            else:
                radius = 1.8
            
            z_proj = np.dot(rel_pos, self.channel_axis)
            xy_proj = rel_pos - z_proj * self.channel_axis
            
            self.atom_data.append({
                'z': z_proj,
                'xy': xy_proj,
                'radius': radius,
                'element': element,
                'residue': atom.get_parent().get_resname()
            })
    
    def identify_pore_region(self):
        """Identify pore region and selectivity filter"""
        # Find conserved sequences
        if self.family_props['pore_characteristics'].get('selectivity_filter'):
            filter_sequences = self.family_props['pore_characteristics']['filter_sequence']
            self.has_filter = self.find_selectivity_filter(filter_sequences)
        else:
            self.has_filter = False
            
    def find_selectivity_filter(self, sequences: List[str]) -> bool:
        """Check for presence of selectivity filter sequence"""
        for chain in self.structure[0]:
            sequence = ''.join(residue.resname for residue in chain)
            for filter_seq in sequences:
                if filter_seq in sequence:
                    return True
        return False
        
    def get_pore_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate pore radius profile along channel axis"""
        z_coords = np.linspace(-20, 20, 41)
        radii = []
        
        for z in z_coords:
            radius = self.get_pore_radius(z)
            radii.append(radius)
            
        return z_coords, np.array(radii)
        
    def get_pore_radius(self, z_coordinate: float) -> float:
        """Calculate pore radius at given z-coordinate"""
        z_tolerance = 2.0
        search_radius = 15.0
        
        nearby_atoms = [
            atom for atom in self.atom_data
            if abs(atom['z'] - z_coordinate) < z_tolerance
        ]
        
        if not nearby_atoms:
            return 0.0
            
        best_radius = 0.0
        coarse_grid = np.linspace(-search_radius, search_radius, 20)
        
        for x in coarse_grid:
            for y in coarse_grid:
                point = np.array([x, y])
                
                max_possible_radius = min(
                    np.linalg.norm(atom['xy'][:2] - point) - atom['radius']
                    for atom in nearby_atoms
                )
                
                if max_possible_radius > best_radius:
                    for dx in np.linspace(-0.5, 0.5, 10):
                        for dy in np.linspace(-0.5, 0.5, 10):
                            refined_point = point + np.array([dx, dy])
                            radius = min(
                                np.linalg.norm(atom['xy'][:2] - refined_point) - atom['radius']
                                for atom in nearby_atoms
                            )
                            best_radius = max(best_radius, radius)
        
        # Apply family-specific corrections
        if self.has_filter and abs(z_coordinate) < 5.0:  # Near selectivity filter
            best_radius *= 0.9  # More restrictive near filter
            
        return max(0.0, best_radius)
        
    def estimate_conductance(self, ion: str) -> float:
        """Estimate conductance for specific ion"""
        ion_props = IonChannelProperties.get_ion_properties(ion)
        if not ion_props:
            raise ValueError(f"Properties not available for ion: {ion}")
            
        # Get pore dimensions
        z_coords, radii = self.get_pore_profile()
        min_radius = min(radii)
        
        logger.info(f"Channel bottleneck radius: {min_radius:.2f} Å")
        logger.info(f"Minimum required radius for {ion}: {ion_props['min_radius']:.2f} Å")
        
        if min_radius < ion_props['min_radius']:
            logger.info(f"Ion {ion} cannot pass due to size constraint")
            return 0.0
            
        # Calculate selectivity factor
        selectivity_sequence = self.family_props['selectivity_sequence']
        try:
            selectivity = 1.0 / (selectivity_sequence.index(ion) + 1)
        except ValueError:
            selectivity = 0.01  # Low selectivity for ions not in sequence
            
        # Base conductance calculation
        base_conductance = self.family_props['base_conductance']
        
        # Physical factors
        area_factor = (min_radius / self.family_props['pore_characteristics']['typical_radius'])**2
        dehydration_factor = np.exp(-ion_props['dehydration_energy'] / 0.5924)  # RT at 310K
        
        # Concentration factors (mM)
        concentrations = {
            'K': 145.0,
            'Na': 12.0,
            'Ca': 0.0001,
            'Cl': 40.0,
            'Mg': 0.5
        }
        
        # Calculate final conductance with channel-specific adjustments
        if self.channel_family == 'Cl':
            # Special handling for CaCC and other chloride channels
            relative_permeability = ion_props.get('relative_permeability', 0.01)
            
            # Enhanced calculation for anion channels
            conductance = (
                base_conductance *
                selectivity *
                area_factor *
                dehydration_factor *
                relative_permeability *
                (concentrations[ion] / concentrations[self.family_props['primary_ion']]) *
                3.0  # Adjustment factor for CaCC channels
            )
        else:
            # Standard calculation for other channels
            conductance = (
                base_conductance *
                selectivity *
                area_factor *
                dehydration_factor *
                (concentrations[ion] / concentrations[self.family_props['primary_ion']])
            )
        
        logger.info(f"Conductance components for {ion}:")
        logger.info(f"Base conductance: {base_conductance*1e12:.1f} pS")
        logger.info(f"Selectivity factor: {selectivity:.3f}")
        logger.info(f"Area factor: {area_factor:.3f}")
        logger.info(f"Dehydration factor: {dehydration_factor:.3f}")
        logger.info(f"Final conductance: {conductance*1e12:.1f} pS")
        
        return conductance
        
    def analyze_multiple_ions(self, ion_list: List[str] = None) -> pd.DataFrame:
        """Analyze conductance for multiple ions"""
        if ion_list is None:
            ion_list = self.family_props['selectivity_sequence']
            
        results = {}
        for ion in ion_list:
            try:
                conductance = self.estimate_conductance(ion)
                results[ion] = f"{conductance*1e12:.1f} pS"
            except Exception as e:
                logger.error(f"Error analyzing {ion}: {str(e)}")
                results[ion] = str(e)
        
        return pd.DataFrame.from_dict(results, orient='index', 
                                    columns=['Conductance'])

if __name__ == "__main__":
    try:
        # Example usage for different channel types
        channel_configs = [
            ("7zdznew.pdb", "CaCC"),
            ("7zdz.pdb", "kir2.1"),
            ("7mjo.pdb", "kir6.1")
        ]
        
        # Analyze the first available channel file
        for pdb_file, channel_name in channel_configs:
            try:
                analyzer = UniversalChannelAnalyzer(pdb_file, channel_name)
                results = analyzer.analyze_multiple_ions()
                print(f"\nConductance Results for {channel_name}:")
                print(results)
                print(f"\nNote: Values are single-channel conductances under physiological conditions")
                break  # Process only the first available file
            except FileNotFoundError:
                continue
            
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")