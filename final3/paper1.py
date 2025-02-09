import numpy as np
import matplotlib.pyplot as plt

class PericyteModel:
    def __init__(self):
        # Time parameters (unchanged)
        self.dt = 0.01
        self.t_end = 1000.0
        self.t = np.arange(0, self.t_end, self.dt)
        
        # Baseline parameters
        self.diameter_baseline = 4.0
        self.cAMP_baseline = 1.0
        
        # α2 receptor parameters
        self.Kd_alpha2 = 0.1
        self.tau_alpha2 = 5.0
        self.max_cAMP_inhibition = 0.85
        
        # Contractility parameters
        self.tau_contraction = 15.0
        self.max_constriction = 0.65  # Increased for ~40% constriction
        self.K_cAMP = 0.5
        
        # α2 antagonist parameters
        self.Ki_atipamezole = 0.01
        
        # Agonist parameters
        self.efficacy_clonidine = 0.95  # Slightly different efficacies
        self.efficacy_xylazine = 0.90
        self.efficacy_phenylephrine = 0  # No effect via α2 pathway
    
    def alpha2_activation(self, NA, atipamezole_conc=0):
        if atipamezole_conc > 0:
            effective_NA = NA * self.Ki_atipamezole / (self.Ki_atipamezole + atipamezole_conc * 10)
        else:
            effective_NA = NA
        return effective_NA / (effective_NA + self.Kd_alpha2)
    
    def compute_cAMP(self, NA, cAMP_prev, atipamezole_conc=0):
        alpha2_act = self.alpha2_activation(NA, atipamezole_conc)
        cAMP_target = self.cAMP_baseline * (1 - self.max_cAMP_inhibition * alpha2_act)
        dcAMP = (cAMP_target - cAMP_prev) * self.dt / self.tau_alpha2
        return cAMP_prev + dcAMP
    
    def compute_diameter(self, cAMP, diameter_prev):
        constriction = self.max_constriction * (1 - cAMP / (cAMP + self.K_cAMP))
        diameter_target = self.diameter_baseline * (1 - constriction)
        ddiameter = (diameter_target - diameter_prev) * self.dt / self.tau_contraction
        return diameter_prev + ddiameter
    
    def run_simulation(self, condition='NA'):
        diameter = np.ones(len(self.t)) * self.diameter_baseline
        cAMP = np.ones(len(self.t)) * self.cAMP_baseline
        NA = np.zeros(len(self.t))
        atipamezole = np.zeros(len(self.t))
        
        stim_start = int(200/self.dt)
        stim_end = int(600/self.dt)
        
        if condition == 'NA':
            NA[stim_start:stim_end] = 2.0
        elif condition == 'NA+atipamezole':
            NA[stim_start:stim_end] = 2.0
            atipamezole[stim_start:stim_end] = 1.0
        elif condition == 'clonidine':
            NA[stim_start:stim_end] = 2.0 * self.efficacy_clonidine
        elif condition == 'xylazine':
            NA[stim_start:stim_end] = 2.0 * self.efficacy_xylazine
        elif condition == 'phenylephrine':
            NA[stim_start:stim_end] = self.efficacy_phenylephrine  # No effect
        
        # Pre-equilibrate
        pre_equil_steps = 10000
        d_temp = self.diameter_baseline
        c_temp = self.cAMP_baseline
        
        for _ in range(pre_equil_steps):
            c_temp = self.compute_cAMP(0, c_temp)
            d_temp = self.compute_diameter(c_temp, d_temp)
        
        diameter[0] = d_temp
        cAMP[0] = c_temp
        
        # Main simulation
        for i in range(1, len(self.t)):
            cAMP[i] = self.compute_cAMP(NA[i], cAMP[i-1], atipamezole[i])
            diameter[i] = self.compute_diameter(cAMP[i], diameter[i-1])
        
        return NA, cAMP, diameter

    def plot_results(self, results_dict):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        colors = {
            'NA': 'blue', 
            'NA+atipamezole': 'red', 
            'clonidine': 'green', 
            'xylazine': 'purple',
            'phenylephrine': 'orange'
        }
        labels = {
            'NA': 'NA', 
            'NA+atipamezole': 'NA+atipamezole',
            'clonidine': 'clonidine', 
            'xylazine': 'xylazine',
            'phenylephrine': 'phenylephrine'
        }
        
        for condition, (na, camp, diam) in results_dict.items():
            axes[0].plot(self.t, na, color=colors[condition], label=labels[condition])
        axes[0].set_ylabel('Drug (µM)')
        axes[0].set_title('Drug Application')
        axes[0].legend()
        
        for condition, (na, camp, diam) in results_dict.items():
            axes[1].plot(self.t, camp, color=colors[condition])
        axes[1].set_ylabel('cAMP (norm.)')
        axes[1].set_title('cAMP Response')
        
        for condition, (na, camp, diam) in results_dict.items():
            axes[2].plot(self.t, diam, color=colors[condition])
        axes[2].set_ylabel('Diameter (µm)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Capillary Diameter')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate constriction percentages
        for condition, (na, camp, diam) in results_dict.items():
            baseline_d = np.mean(diam[0:int(200/self.dt)])
            min_d = np.min(diam)
            constriction = (baseline_d - min_d) / baseline_d * 100
            print(f"{condition} constriction: {constriction:.1f}%")

# Run simulations
model = PericyteModel()
results = {}

conditions = ['NA', 'NA+atipamezole', 'clonidine', 'xylazine', 'phenylephrine']
for condition in conditions:
    results[condition] = model.run_simulation(condition)

model.plot_results(results)