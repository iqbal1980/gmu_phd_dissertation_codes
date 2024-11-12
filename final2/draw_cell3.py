import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arrow

def create_simplified_cell_schematic():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Cell membrane
    cell = Circle((5, 5), 4, fill=False, color='black')
    ax.add_patch(cell)

    # ER
    er = Circle((5, 5), 2, fill=False, color='blue')
    ax.add_patch(er)

    # Subspace
    subspace = Circle((5, 5), 3.8, fill=False, color='lightgray', linestyle='--')
    ax.add_patch(subspace)

    # Channels and pumps
    channel_positions = {
        'KIR6.1': (3, 8.5), 'KIR2.2': (7, 8.5),
        'TRPC1': (1.5, 5), 'TRPC3': (8.5, 5),
        'CaCC': (3, 1.5), 'ClC-2': (7, 1.5),
        'L-type': (5, 9),
        'IP3R': (5, 7), 'SERCA': (5, 3)
    }

    for channel, pos in channel_positions.items():
        rect = Rectangle((pos[0]-0.3, pos[1]-0.3), 0.6, 0.6, fill=True, color='red')
        ax.add_patch(rect)
        ax.text(pos[0], pos[1]-0.6, channel, ha='center', va='top', fontsize=8)

    # Fluxes
    ax.add_patch(Arrow(5, 9, 0, -0.5, width=0.3, color='green'))  # L-type Ca2+ influx
    ax.add_patch(Arrow(5, 7, 0, -1, width=0.3, color='orange'))  # IP3R Ca2+ release
    ax.add_patch(Arrow(5, 3, 0, 1, width=0.3, color='purple'))  # SERCA Ca2+ uptake
    ax.add_patch(Arrow(5, 8.3, 0, -0.5, width=0.3, color='gray', linestyle='--'))  # Subspace to cytosol diffusion

    # Labels
    ax.text(5, 9.5, 'Extracellular space', ha='center', va='bottom')
    ax.text(5, 0.5, 'Cytoplasm', ha='center', va='top')
    ax.text(5, 5, 'ER', ha='center', va='center', color='blue')
    ax.text(8, 8, 'Subspace', ha='center', va='center', color='gray')

    plt.title('Simplified Pericyte Cell Schematic')
    plt.tight_layout()
    plt.show()

create_simplified_cell_schematic()