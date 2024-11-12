import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arrow, Wedge

def create_cell_schematic():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Cell membrane
    cell = Circle((5, 5), 4.5, fill=False, color='black')
    ax.add_patch(cell)

    # ER
    er = Wedge((5, 5), 3, 45, 315, width=0.5, fill=False, color='blue')
    ax.add_patch(er)

    # Mitochondria
    mito = Circle((3, 6), 1, fill=False, color='green')
    ax.add_patch(mito)

    # Subspace
    subspace = Wedge((5, 5), 4.5, 0, 360, width=0.3, fill=True, color='lightgray', alpha=0.5)
    ax.add_patch(subspace)

    # Channels and pumps
    channel_positions = {
        'CRAC': (5, 9.5),
        'IP3R': (6.5, 6.5),
        'SERCA': (3.5, 3.5),
        'PMCA': (1, 5),
        'MCU': (3.5, 6.5),
        'mNCX': (2.5, 5.5)
    }

    for channel, pos in channel_positions.items():
        rect = Rectangle((pos[0]-0.3, pos[1]-0.3), 0.6, 0.6, fill=True, color='red')
        ax.add_patch(rect)
        ax.text(pos[0], pos[1]-0.6, channel, ha='center', va='top', fontsize=8)

    # Fluxes
    ax.add_patch(Arrow(5, 9.5, 0, -0.5, width=0.3, color='orange'))  # CRAC influx
    ax.add_patch(Arrow(6.5, 6.5, -0.5, -0.5, width=0.3, color='red'))  # IP3R release
    ax.add_patch(Arrow(3.5, 3.5, 0.5, 0.5, width=0.3, color='blue'))  # SERCA uptake
    ax.add_patch(Arrow(1, 5, -0.5, 0, width=0.3, color='purple'))  # PMCA efflux
    ax.add_patch(Arrow(3.5, 6.5, -0.2, -0.2, width=0.3, color='green'))  # MCU uptake
    ax.add_patch(Arrow(2.5, 5.5, 0.2, 0.2, width=0.3, color='brown'))  # mNCX efflux
    ax.add_patch(Arrow(5, 8.5, 0, -0.5, width=0.3, color='gray', linestyle='--'))  # Subspace to cytosol diffusion

    # Labels
    ax.text(5, 9.8, 'Extracellular space', ha='center', va='bottom')
    ax.text(5, 0.2, 'Cytoplasm', ha='center', va='bottom')
    ax.text(5, 5, 'ER', ha='center', va='center', color='blue')
    ax.text(3, 6, 'Mito', ha='center', va='center', color='green')
    ax.text(8, 8, 'Subspace', ha='center', va='center', color='gray')

    plt.title('Pericyte Cell Schematic Based on Fortran Model')
    plt.tight_layout()
    plt.show()

create_cell_schematic()