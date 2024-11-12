import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class TCellModel:
    def __init__(self):
        # Constants and parameters
        self.v1 = 90.0
        self.ak3 = 0.1
        self.v2 = 0.00
        self.v3 = 1.0
        
        # IP3R parameters
        self.a1 = 400.0
        self.a3 = 400.0
        self.a4 = 0.2
        self.d1 = 0.13
        self.d3 = 0.9434
        self.d4 = 0.1445
        self.a2 = 0.2
        self.a5 = 20.0
        self.d2 = 1.049
        self.d5 = 82.34e-3
        self.c2 = 0.2
        
        # Buffer parameters
        self.Bscyt = 225.0
        self.aKscyt = 0.1
        self.Bscas = 5000.0
        self.aKscas = 0.1
        self.Bm = 111.0
        self.aKm = 0.123
        self.Bser = 2000.0
        self.aKser = 1.0
        
        # Universal constants
        self.F = 96500.0
        self.T = 310.0
        self.R = 8314.0
        self.z = 2.0
        self.Cm = 2.5e-3
        
        # Ion channel parameters
        self.gna = 3.0
        self.gt = 3.0
        self.Ek = -84.0
        self.Eca = 60.0
        self.Emax = 0.8
        self.akd = 0.45
        self.ax = 3.0
        self.gtrpc3 = 0.00195
        self.aKmca = 0.6
        self.PcaL = 0.3e-7
        
        # I_crac parameters
        self.taoinact = 100.0
        self.taoact = 3.0
        self.Pca = 2.8e-10
        self.alphi = 1.0
        self.alpho = 0.341
        self.akk = 200.0
        self.akj = 1.0
        
        # PMCA parameters
        self.vu = 1.0 * 1540000.0
        self.vm = 1.0 * 2200000.0
        self.aku = 0.303
        self.akmp = 0.14
        self.aru = 1.8
        self.arm = 2.1
        
        # IP3 calculation parameters
        self.V2ip3 = 12.5
        self.ak2ip3 = 6.0
        self.V3ip3 = 0.9
        self.ak3ip3 = 0.1
        self.ak4ip3 = 1.0
        
        self.taos = 0.5
        
        # Mitochondria parameters
        self.psi = 160.0
        self.alphm = 0.2
        self.Pmito = 2.776e-14
        self.Vnc = 1.836
        self.aNa = 5000.0
        self.akna = 8000.0
        self.akca = 8.0
        
        # Cell Geometry
        self.Vt = 2.0e-12
        self.Vcyto = self.Vt * 0.55
        self.Ver = self.Vcyto * 0.2
        self.Vmito = self.Vcyto * 0.08
        self.Vss = self.Vcyto * 0.1
        self.c1 = self.Ver / self.Vcyto
        self.c3 = self.Vmito / self.Vss
        self.c4 = self.Vcyto / self.Vss
        self.c5 = self.Vmito / self.Vcyto
        
        # Derived parameters
        self.b1 = self.a1 * self.d1
        self.b2 = self.a2 * self.d2
        self.b3 = self.a3 * self.d3
        self.b4 = self.a4 * self.d4
        self.b5 = self.a5 * self.d5

        # Initialize NFAT, NFkB, and JNK modules
        self.nfat = NFATModule()
        self.nfkb = NFkBModule()
        self.jnk = JNKModule()
        
        self.epsilon = 1e-10
        self.max_cai = 1e6  # Set a reasonable maximum value for cai

    def initialize_state(self):
        # Combine all state variables into a single array
        state = [
            0.1, 300.0, 0.1, 0.01, 0.01, 0.1, 0.1,  # cai, caer, cam, aip3, dag, x100, x110
            1.0, 0.1, -70.0, 0.02, 1.0, 0.0, 1.0,   # dpmca, cas, V, an, aj, ad, af
            0.0, 1.0, 0.0,                          # aa5, ah5, xCai
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,           # x000, x010, x001, x101, x011, x111
            1.0, 0.0                                # proinact, proact
        ]
        state.extend(self.nfat.x)
        state.extend(self.nfkb.x)
        state.extend(self.jnk.x)
        return np.array(state)

 
    def derivatives(self, t, state):
        # Unpack the state vector
        (self.cai, self.caer, self.cam, self.aip3, self.dag, self.x100, self.x110,
         self.dpmca, self.cas, self.V, self.an, self.aj, self.ad, self.af,
         self.aa5, self.ah5, self.xCai,
         self.x000, self.x010, self.x001, self.x101, self.x011, self.x111,
         self.proinact, self.proact) = state[:25]
        
        nfat_start = 25
        nfkb_start = nfat_start + len(self.nfat.x)
        jnk_start = nfkb_start + len(self.nfkb.x)
        
        self.nfat.x = state[nfat_start:nfkb_start]
        self.nfkb.x = state[nfkb_start:jnk_start]
        self.jnk.x = state[jnk_start:]

        # Calculate all the rates of change
        dx = np.zeros_like(state)

        # IP3 concentration calculation
        if t < 1200.0:
            cao = 2000.0
        else:
            cao = 0.0
        if t < 200.0 or t > 1200:
            prodip3 = 0.01
        else:
            prodip3 = 5.0

        dx[3] = prodip3 - self.V2ip3 / (1.0 + (self.ak2ip3 / self.aip3)) - \
                (self.V3ip3 / (1 + (self.ak3ip3 / self.aip3))) * \
                (1.0 / (1.0 + (self.ak4ip3 / self.cai)))
        dx[4] = prodip3 - 1.0 * self.dag

        # Buffer calculations
        buer = 1.0 / (1.0 + self.Bser * self.aKser / ((self.aKser + self.caer)**2) + 
                      (self.Bm * self.aKm) / ((self.aKm + self.caer)**2))
        bucyt = 1.0 / (1 + (self.Bscyt * self.aKscyt) / ((self.aKscyt + self.cai)**2) + 
                       (self.Bm * self.aKm) / ((self.aKm + self.cai)**2))
        bucas = 1.0 / (1 + (self.Bscas * self.aKscas) / ((self.aKscas + self.cas)**2) + 
                       (self.Bm * self.aKm) / ((self.aKm + self.cas)**2))

        # IP3R calculations
        fchan = self.c1 * (self.v1 * (self.x110**3)) * (self.caer - self.cai)
        fleak = self.c1 * self.v2 * (self.caer - self.cai)
        fpump = self.v3 * self.cai**2 / (self.ak3**2 + self.cai**2)

        # Mitochondrial calculations
        if self.psi == 0:
            cJuni = (self.Pmito / self.Vmito) * (self.alphm * self.cam - self.alphi * self.cas)
        else:
            bb = (self.z * self.psi * self.F) / (self.R * self.T)
            cJuni = (self.Pmito / self.Vmito) * bb * ((self.alphm * self.cam * np.exp(-bb) - 
                     self.alphi * self.cas) / (np.exp(-bb) - 1))

        som = self.aNa**3 * self.cam / (self.akna**3 * self.akca)
        soe = self.aNa**3 * self.cas / (self.akna**3 * self.akca)
        cJnc = self.Vnc * (np.exp(0.5 * self.psi * self.F / (self.R * self.T)) * som - 
                           np.exp(-0.5 * self.psi * self.F / (self.R * self.T)) * soe / 
                           (1 + self.aNa**3 / (self.akna**3) + self.cam / self.akca + som + 
                            self.aNa**3 / (self.akna**3) + self.cas / self.akca + soe))
        dx[2] = 0.01 * (cJuni - cJnc)

        # Current calculations
        # I_crac
        if self.V == 0:
            currca = self.Pca * self.z * self.F * (self.alphi * self.cas - self.alpho * cao)
        else:
            B = (self.z * self.V * self.F) / (self.R * self.T)
            currca = self.Pca * self.z * self.F * B * ((self.alphi * self.cas * np.exp(B) - self.alpho * cao) / 
                                                       (np.exp(B) - 1))
        cfinf = (self.akj**2 / (self.akj**2 + self.cas**2))
        cdinf = (self.akk**4.7 / (self.akk**4.7 + self.caer**4.7))
        self.taoinact = 200 * (2.0**2 / (2.0**2 + self.cas**2))
        dx[23] = (cfinf - self.proinact) / self.taoinact
        dx[24] = (cdinf - self.proact) / self.taoact
        procrac = self.proact * self.proinact
        currca *= procrac
        cJcrac = -currca / (1e6 * self.Vss * self.z * self.F)

        # L-type Ca2+ current
        if self.V == 0:
            barIca = self.PcaL * self.z * self.F * (self.alphi * self.cai - self.alpho * cao)
        else:
            B = (self.z * self.V * self.F) / (self.R * self.T)
            barIca = self.PcaL * self.z * self.F * B * ((self.alphi * self.cai * np.exp(B) - self.alpho * cao) / 
                                                        (np.exp(B) - 1))
        afca = 1.0 / (1.0 + (self.cai / self.aKmca)**2)
        adinf = 1.0 / (1.0 + np.exp(-(self.V + 10.0) / 6.24))
        afinf = 1.0 / (1.0 + np.exp((self.V + 35.06) / 8.6)) + 0.6 / (1 + np.exp((50 - self.V) / 20.0))
        taoad = adinf * (1.0 - np.exp(-(self.V + 10.0) / 6.24)) / (0.035 * (self.V + 10.0))
        taoaf = 1.0 / (0.0197 * np.exp(-(0.0337 * (self.V + 10.0))**2) + 0.02)
        dx[12] = (adinf - self.ad) / taoad
        dx[13] = (afinf - self.af) / taoaf
        aIca = self.ad * self.af * afca * barIca
        cJca = -aIca / (1e6 * self.Vcyto * self.z * self.F)

        # K+ channel
        aninf = 1.0 / (1.0 + np.exp((self.V + 11.0) / (-15.2)))
        ajinf = 1.0 / (1.0 + np.exp((self.V + 45.0) / 9.0))
        taoan = 1.0 / (0.2 * np.exp(0.032 * (self.V + 1.42)))
        taoaj = 15.0 / (0.03 * (np.exp(0.0083 * (self.V + 40.8)) + 
                                0.4865 * np.exp(-0.06 * (self.V + 60.49))))
        dx[10] = (aninf - self.an) / taoan
        dx[11] = (ajinf - self.aj) / taoaj
        aIt = self.gt * self.an * self.aj * (self.V - self.Ek)

        # Ca2+ activated K+
        E = self.Emax / (1 + (self.akd / self.cai)**self.ax)
        aIkca = E * (self.V - self.Ek)
        
        # Na+ current
        aa3 = 1.0 / (1.0 + np.exp((self.V + 51.9005) / (-3.1011)))
        aa4 = 0.3959 * np.exp(-0.017 * self.V)
        dx[14] = (aa3 - self.aa5) / aa4
        ah3 = 1.0 / (1.0 + np.exp((self.V + 76.5769) / 7.1485))
        ah4 = 2.9344 * np.exp(-0.0077 * self.V)
        dx[15] = (ah3 - self.ah5) / ah4
        Ena = 30.0
        aIna = self.gna * self.aa5**3 * self.ah5 * (self.V - Ena)

        # Safely calculate u4 and u5
        safe_cai = max(self.cai, self.epsilon)
        safe_cai = min(safe_cai, self.max_cai)  # Limit cai to prevent overflow
        self.u4 = (self.vu * (safe_cai**self.aru) / 
                   (safe_cai**self.aru + self.aku**self.aru)) / (6.6253 * 1e5)
        self.u5 = (self.vm * (safe_cai**self.arm) / 
                   (safe_cai**self.arm + self.akmp**self.arm)) / (6.6253 * 1e5)


        w1 = 0.1 * self.cai
        w2 = 0.01
        taom = 1 / (w1 + w2)
        dpmcainf = w2 / (w1 + w2)
        dx[7] = (dpmcainf - self.dpmca) / taom
        cJpmca = (self.dpmca * self.u4 + (1 - self.dpmca) * self.u5)
        aIpmca = cJpmca * self.z * self.F * self.Vcyto * 1e6

        # TRPM4 channel
        Etrpm4 = 0.2
        gtrpm4 = 1.2
        
        # Safely calculate xCai_inf
        xCai_inf = 1.0 / (1.0 + max((safe_cai / 1.3), self.epsilon)**(-1.1))
        
        xv = 0.05 + (0.95 / (1.0 + np.exp(-(self.V + 40.0) / 15.0)))
        taoxcai = 30.0
        axC = xCai_inf / taoxcai
        bxC = (1.0 - xCai_inf) / taoxcai
        dx[16] = axC * (1.0 - self.xCai) - bxC * self.xCai
        aItrpm4 = gtrpm4 * self.xCai * xv * (self.V - Etrpm4)
        gNatrpm4 = aItrpm4 * (Etrpm4 - Ena) / ((self.V - Ena) * (self.Ek - Etrpm4) + (self.V - self.Ek) * (Etrpm4 - Ena))
        gKtrpm4 = aItrpm4 * (-Etrpm4 + self.Ek) / ((self.V - Ena) * (self.Ek - Etrpm4) + (self.V - self.Ek) * (Etrpm4 - Ena))
        aINatrpm4 = gNatrpm4 * (self.V - Ena)
        aIKtrpm4 = gKtrpm4 * (self.V - self.Ek)
        


        # TRPC3
        if cao == 0:
            aItrpc3 = 0.0
            self.Eca = 0
        else:
            self.Eca = (self.R * self.T / (self.z * self.F)) * np.log(cao / safe_cai)
            aItrpc3 = self.gtrpc3 * (self.dag / (self.dag + 2.0)) * (self.V - self.Eca)

 

            
            
        cJtrpc3 = -aItrpc3 / (1e6 * self.Vcyto * self.z * self.F)



        # Background Ca current Cab
        gcab = 0.0
        aIcab = gcab * (self.V - self.Eca)
        cJcab = -aIcab / (1e6 * self.Vcyto * self.z * self.F)

        # Cl channel
        Ecl = -33.0
        gcl = 70e-3
        aIcl = gcl * (self.V - Ecl)

        # Membrane potential
        dx[9] = (-aIca - aIt - aIkca - currca - aIpmca - aIna - aItrpc3 - aItrpm4 - aIcl) / self.Cm

        # Update calcium concentrations
        dx[1] = buer * (fpump - fchan - fleak) / self.c1
        dx[8] = bucas * (self.c3 * (cJnc - cJuni) + self.c4 * cJcrac - 
                         self.c4 * (self.cas - self.cai) / self.taos)
        dx[0] = bucyt * (fchan + fleak - fpump - cJpmca + 
                         (self.cas - self.cai) / self.taos + cJca + cJtrpc3 + cJcab)

        # Update IP3R states
        f1 = self.b5 * self.x010 - self.a5 * self.cai * self.x000
        f2 = self.b1 * self.x100 - self.a1 * self.aip3 * self.x000
        f3 = self.b4 * self.x001 - self.a4 * self.cai * self.x000
        f4 = self.b5 * self.x110 - self.a5 * self.cai * self.x100
        f5 = self.b2 * self.x101 - self.a2 * self.cai * self.x100
        f6 = self.b1 * self.x110 - self.a1 * self.aip3 * self.x010
        f7 = self.b4 * self.x011 - self.a4 * self.cai * self.x010
        f8 = self.b5 * self.x011 - self.a5 * self.cai * self.x001
        f9 = self.b3 * self.x101 - self.a3 * self.aip3 * self.x001
        f10 = self.b2 * self.x111 - self.a2 * self.cai * self.x110
        f11 = self.b5 * self.x111 - self.a5 * self.cai * self.x101
        f12 = self.b3 * self.x111 - self.a3 * self.aip3 * self.x011

        dx[17] = -(f1 + f2 + f3)
        dx[5] = f4 + f5 - f2
        dx[18] = -f1 + f6 + f7
        dx[19] = f8 - f3 + f9
        dx[6] = -f4 - f6 + f10
        dx[20] = f11 - f9 - f5
        dx[21] = -f8 - f7 + f12
        dx[22] = -f11 - f12 - f10

        # Update NFAT, NFkB, and JNK modules
        dx[nfat_start:nfkb_start] = self.nfat.derivatives(self.cai)
        dx[nfkb_start:jnk_start] = self.nfkb.derivatives(self.cai)
        dx[jnk_start:] = self.jnk.derivatives(self.cai)

        return dx

    def run_simulation(self, total_time, dt):
        initial_state = self.initialize_state()
        
 
        
        # Use a variable step-size solver with more lenient error tolerances
        sol = solve_ivp(
            fun=self.derivatives,
            t_span=(0, total_time),
            y0=initial_state,
            method='Radau',
            t_eval=np.arange(0, total_time, dt),
            rtol=1e-3,  # Increased from 1e-6
            atol=1e-6,  # Increased from 1e-8
            max_step=dt,
            first_step=dt/100  # Start with a smaller first step
        )

        # Prepare results storage
        results = {
            'time': sol.t,
            'cai': sol.y[0],
            'caer': sol.y[1],
            'cam': sol.y[2],
            'cas': sol.y[8],
            'V': sol.y[9],
            'aip3': sol.y[3],
            'dag': sol.y[4],
            'x100': sol.y[5],
            'x110': sol.y[6],
            'nfat_active': np.zeros_like(sol.t),
            'nfkb_active': np.zeros_like(sol.t),
            'jnk_active': np.zeros_like(sol.t),
        }

        nfat_start = 25
        nfkb_start = nfat_start + len(self.nfat.x)
        jnk_start = nfkb_start + len(self.nfkb.x)

        # Calculate active fractions
        for i in range(len(sol.t)):
            results['nfat_active'][i] = self.nfat.get_active_fraction(sol.y[nfat_start:nfkb_start, i])
            results['nfkb_active'][i] = self.nfkb.get_active_fraction(sol.y[nfkb_start:jnk_start, i])
            results['jnk_active'][i] = self.jnk.get_active_fraction(sol.y[jnk_start:, i])

        return results

    def plot_results(self, results):
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle('T-Cell Model Simulation Results')

        axs[0, 0].plot(results['time'], results['cai'])
        axs[0, 0].set_ylabel('cai (μM)')
        axs[0, 0].set_xlabel('Time (s)')

        axs[0, 1].plot(results['time'], results['caer'])
        axs[0, 1].set_ylabel('caer (μM)')
        axs[0, 1].set_xlabel('Time (s)')

        axs[1, 0].plot(results['time'], results['cam'])
        axs[1, 0].set_ylabel('cam (μM)')
        axs[1, 0].set_xlabel('Time (s)')

        axs[1, 1].plot(results['time'], results['cas'])
        axs[1, 1].set_ylabel('cas (μM)')
        axs[1, 1].set_xlabel('Time (s)')

        axs[2, 0].plot(results['time'], results['V'])
        axs[2, 0].set_ylabel('V (mV)')
        axs[2, 0].set_xlabel('Time (s)')

        axs[2, 1].plot(results['time'], results['aip3'])
        axs[2, 1].set_ylabel('IP3 (μM)')
        axs[2, 1].set_xlabel('Time (s)')

        axs[3, 0].plot(results['time'], results['nfat_active'])
        axs[3, 0].set_ylabel('NFAT Active Fraction')
        axs[3, 0].set_xlabel('Time (s)')

        axs[3, 1].plot(results['time'], results['nfkb_active'])
        axs[3, 1].set_ylabel('NFkB Active Fraction')
        axs[3, 1].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

class NFATModule:
    def __init__(self):
        # NFAT parameters
        self.ak1 = 0.0000256
        self.ak2 = 0.00256
        self.ak3 = 0.005
        self.ak4 = 0.5
        self.ak5 = 0.0019
        self.ak6 = 0.00092
        self.ak7 = 0.005
        self.ak8 = 0.5
        self.ak9 = 0.5
        self.ak10 = 0.005
        self.ak11 = 6.63
        self.ak12 = 0.00168
        self.ak13 = 0.5
        self.ak14 = 0.00256
        self.ak15 = 0.00168
        self.ak16 = 6.63
        self.ak17 = 0.0015
        self.ak18 = 0.00096
        self.ak19 = 1.0
        self.ak20 = 1.0
        self.ak21 = 0.21
        self.ak22 = 0.5

        self.volc = 269
        self.voln = 113
        self.vratio = self.volc / self.voln
        self.vratioc = self.voln / self.volc

        # Initialize NFAT state variables
        self.x = np.zeros(14)
        self.x[0] = 5.21909E-04  # [NFAT]n (μM)
        self.x[1] = 1.10101E-04  # [NFAT]c (μM)
        self.x[2] = 5.05386E-05  # [C*]n (μM)
        self.x[3] = 9.14784E-06  # [C*]c (μM)
        self.x[4] = 2.27232E-04  # [NFAT:Pi]n (μM)
        self.x[5] = 9.43972E-03  # [NFAT:Pi]c (μM)
        self.x[6] = 2.52431E-06  # [NFAT:Pi:C*]n (μM)
        self.x[7] = 2.20743E-06  # [NFAT:Pi:C*]c (μM)
        self.x[8] = 9.47710E-04  # [NFAT:C*]n (μM)
        self.x[9] = 6.14972E-06  # [NFAT:C*]c (μM)
        self.x[10] = 4.91984E-02  # [C]n (μM)
        self.x[11] = 9.71085E-03  # [C]c (μM)

    def derivatives(self, cai):
        dx = np.zeros(14)
        
        # Update calcium concentrations
        self.x[12] = cai  # [Ca]n (μM)
        self.x[13] = cai  # [Ca]c (μM)

        dx[0] = (self.ak1 * self.x[4] - self.ak2 * self.x[0] + self.ak17 * self.x[1] * self.vratio - 
                 self.ak18 * self.x[0] + self.ak15 * self.x[8] - self.ak16 * self.x[0] * self.x[2])

        dx[1] = (self.ak1 * self.x[5] - self.ak2 * self.x[1] + self.ak18 * self.x[0] * self.vratioc - 
                 self.ak17 * self.x[1] + self.ak15 * self.x[9] - self.ak16 * self.x[1] * self.x[3])

        eki = 20 * (self.x[12]**2 / (self.x[12]**2 + 1**2)) * (self.x[12]**2 / (self.x[12]**2 + 10**2))
        dx[2] = (-self.ak11 * self.x[4] * self.x[2] + self.ak12 * self.x[6] + self.ak5 * self.x[3] * self.vratio - 
                 self.ak6 * self.x[2] + self.ak15 * self.x[8] - self.ak16 * self.x[2] * self.x[0] + 
                 self.ak19 * self.x[10] * eki - self.ak20 * self.x[2])

        eki = 20 * (self.x[13]**2 / (self.x[13]**2 + 1**2)) * (self.x[13]**2 / (self.x[13]**2 + 10**2))
        dx[3] = (-self.ak11 * self.x[5] * self.x[3] + self.ak12 * self.x[7] - self.ak5 * self.x[3] + 
                 self.ak6 * self.x[2] * self.vratioc + self.ak15 * self.x[9] - self.ak16 * self.x[3] * self.x[1] + 
                 self.ak19 * self.x[11] * eki - self.ak20 * self.x[3])

        dx[4] = (-self.ak1 * self.x[4] + self.ak2 * self.x[0] - self.ak4 * self.x[4] + 
                 self.ak3 * self.x[5] * self.vratio - self.ak11 * self.x[4] * self.x[2] + self.ak12 * self.x[6])

        dx[5] = (-self.ak1 * self.x[5] + self.ak2 * self.x[1] + self.ak4 * self.x[4] * self.vratioc - self.ak3 * self.x[5] - 
                 self.ak11 * self.x[5] * self.x[3] + self.ak12 * self.x[7])

        dx[6] = (self.ak11 * self.x[4] * self.x[2] - self.ak12 * self.x[6] + self.ak7 * self.x[7] * self.vratio - 
                 self.ak8 * self.x[6] - self.ak13 * self.x[6] + self.ak14 * self.x[8])

        dx[7] = (self.ak11 * self.x[5] * self.x[3] - self.ak12 * self.x[7] + self.ak8 * self.x[6] * self.vratioc - 
                 self.ak7 * self.x[7] - self.ak13 * self.x[7] + self.ak14 * self.x[9])

        dx[8] = (self.ak16 * self.x[0] * self.x[2] - self.ak15 * self.x[8] + self.ak13 * self.x[6] - self.ak14 * self.x[8] + 
                 self.ak9 * self.x[9] * self.vratio - self.ak10 * self.x[8])

        dx[9] = (self.ak16 * self.x[1] * self.x[3] - self.ak15 * self.x[9] + self.ak13 * self.x[7] - self.ak14 * self.x[9] + 
                 self.ak10 * self.x[8] * self.vratioc - self.ak9 * self.x[9])

        eki = 20 * (self.x[12]**2 / (self.x[12]**2 + 1**2)) * (self.x[12]**2 / (self.x[12]**2 + 10**2))
        dx[10] = (self.ak5 * self.x[11] * self.vratio - self.ak6 * self.x[10] - self.ak19 * self.x[10] * eki + 
                  self.ak20 * self.x[2])

        eki = 20 * (self.x[13]**2 / (self.x[13]**2 + 1**2)) * (self.x[13]**2 / (self.x[13]**2 + 10**2))
        dx[11] = (-self.ak5 * self.x[11] + self.ak6 * self.x[10] * self.vratioc - self.ak19 * self.x[11] * eki + 
                  self.ak20 * self.x[3])

        return dx

    def get_active_fraction(self, x):
        return x[8] * self.voln * 6.023e2 / ((x[1] + x[5] + x[7] + x[9]) * self.volc * 6.023e2 + 
                                             (x[0] + x[4] + x[6] + x[8]) * self.voln * 6.023e2)

class NFkBModule:
    def __init__(self):
        # NFkB parameters
        self.ak1 = 0.000614
        self.ak2 = 0.00184
        self.ak3 = 0.0020
        self.ak4 = 0.0010
        self.ak5 = 0.00026
        self.ak6 = 0.0134
        self.ak7 = 0.010
        self.ak8 = 0.02
        self.ak9 = 0.000034
        self.ak10 = 0.000034
        self.ak11 = 0.02
        self.ak12 = 0.000034
        self.ak13 = 0.00092
        self.ak14 = 0.0019
        self.ak15 = 1.0
        self.ak16 = 0.000000001
        self.ak17 = 0.00092
        self.ak18 = 0.0019
        self.ak19 = 0.5
        self.ak20 = 0.21
        self.ak21 = 0.02
        self.ak22 = 0.0022 / 12.4
        self.ak23 = 0.000000036
        self.ak24 = 0.00008
        self.ak25 = 0.0000016
        self.ak26 = 0.0008
        self.ak27 = 0.0016
        self.ak28 = 0.0006
        self.ak29 = 0.009
        self.ak30 = 0.02
        self.tr2a = 0.00154
        self.tr2 = 0.0000165
        self.tr3 = 0.00028

        self.volc = 269
        self.voln = 113
        self.vratio = self.volc / self.voln
        self.vratioc = self.voln / self.volc

        # Initialize NFkB state variables
        self.x = np.zeros(19)
        self.x[0] = 7.17792E+00  # [NFkB]n (nM)
        self.x[1] = 7.31124E-02  # [NFkB]c (nM)
        self.x[2] = 2.03563E-01  # [IkB]n (nM)
        self.x[3] = 5.25426E-02  # [IkB]c (nM)
        self.x[4] = 2.34342E-01  # [NFkB:IkB:Pi]n (nM)
        self.x[5] = 1.07661E-01  # [NFkB:IkB:Pi]c (nM)
        self.x[6] = 2.37664E-01  # [NFkB:IkB]n (nM)
        self.x[7] = 5.29613E+01  # [NFkB:IkB]c (nM)
        self.x[8] = 5.05386E-02  # [C*]n (nM)
        self.x[9] = 9.14784E-03  # [C*]c (nM)
        self.x[10] = 4.91984E+01  # [C]n (nM)
        self.x[11] = 9.71085E+00  # [C]c (nM)
        self.x[14] = 0.0  # [IKK*:PKCheta] (%)
        self.x[15] = 0.0  # [IKK*:PKCalpha:Cn] (%)
        self.x[16] = 0.0  # [IKK*]c (%)
        self.x[17] = 0.0  # [IKK*]n (%)
        self.x[18] = 0.0  # IkBt

        self.PKCtheta = 2000.0
        self.Ptotal = 1000.0

    def derivatives(self, cai):
        dx = np.zeros(19)
        
        # Update calcium concentrations
        self.x[12] = cai * 1000  # [Ca]n (nM)
        self.x[13] = cai * 1000  # [Ca]c (nM)

        dx[0] = (-self.ak5 * self.x[0] + self.ak6 * self.x[1] * self.vratio - 
                 self.ak1 * self.x[0] * self.x[2] + self.ak2 * self.x[4])
        dx[1] = (-self.ak6 * self.x[1] + self.ak5 * self.x[0] * self.vratioc - 
                 self.ak1 * self.x[1] * self.x[3] + self.ak2 * self.x[5])
        dx[2] = (-self.ak7 * self.x[2] + self.ak8 * self.x[3] * self.vratio - 
                 self.ak1 * self.x[0] * self.x[2] + self.ak2 * self.x[4])
        dx[3] = (-self.ak8 * self.x[3] + self.ak7 * self.x[2] * self.vratioc - 
                 self.ak1 * self.x[1] * self.x[3] + self.ak2 * self.x[5] - 
                 self.ak21 * self.x[3] + self.ak22 * self.x[18])
        dx[4] = (self.ak1 * self.x[0] * self.x[2] - self.ak2 * self.x[4] - self.ak3 * self.x[4] + 
                 self.ak4 * self.x[6] * self.x[17] + self.ak10 * self.x[5] * self.vratio - self.ak9 * self.x[4])
        dx[5] = (self.ak1 * self.x[1] * self.x[3] - self.ak3 * self.x[5] + self.ak4 * self.x[7] * self.x[16] - 
                 self.ak2 * self.x[5] + self.ak9 * self.x[4] * self.vratioc - self.ak10 * self.x[5])
        dx[6] = (self.ak3 * self.x[4] + self.ak12 * self.x[7] * self.vratio - self.ak11 * self.x[6] - 
                 self.ak4 * self.x[6] * self.x[17])
        dx[7] = (self.ak3 * self.x[5] - self.ak4 * self.x[7] * self.x[16] + 
                 self.ak11 * self.x[6] * self.vratioc - self.ak12 * self.x[7])
        dx[8] = (-self.ak15 * self.x[8] + self.ak16 * self.x[12]**3 * self.x[10] - 
                 self.ak13 * self.x[8] + self.ak14 * self.x[9] * self.vratio)
        dx[9] = (-self.ak15 * self.x[9] + self.ak16 * self.x[13]**3 * self.x[11] + 
                 self.ak13 * self.x[8] * self.vratioc - self.ak14 * self.x[9])
        dx[10] = (self.ak15 * self.x[8] - self.ak16 * self.x[10] * self.x[12]**3 + 
                  self.ak18 * self.x[11] * self.vratio - self.ak17 * self.x[10])
        dx[11] = (self.ak15 * self.x[9] - self.ak16 * self.x[11] * self.x[13]**3 - 
                  self.ak18 * self.x[11] + self.ak17 * self.x[10] * self.vratioc)
        dx[14] = (self.ak23 * (1 - self.x[14]) * self.PKCtheta - self.ak24 * self.x[14])
        dx[15] = (self.ak25 * (1 - self.x[15]) * self.Ptotal - self.ak26 * self.x[15] + 
                  self.ak27 * (1 - self.x[15]) * self.x[9] - self.ak28 * self.x[15])
        dx[16] = (dx[14] + dx[15] + self.ak30 * self.x[17] * self.vratioc - self.ak29 * self.x[16])
        dx[17] = (self.ak29 * self.x[16] * self.vratio - self.ak30 * self.x[17])
        dx[18] = (self.tr2a + self.tr2 * self.x[0] - self.tr3 * self.x[18])

        return dx

    def get_active_fraction(self, x):
        return x[0] * self.vratioc / ((x[0] + x[4] + x[6]) * self.voln + 
                                      (x[1] + x[5] + x[7]) * self.volc) * self.volc

class JNKModule:
    def __init__(self):
        # JNK parameters
        self.ak13 = 0.00092
        self.ak14 = 0.0019
        self.ak15 = 1.0
        self.ak16 = 0.000000001
        self.ak17 = 0.00092
        self.ak18 = 0.0019
        self.ak19 = 0.5
        self.ak20 = 0.21
        self.ak35 = 0.016
        self.ak36 = 0.00038
        self.ak41 = 0.00001
        self.ak42 = 0.00025
        self.ak37 = 0.00012
        self.ak38 = 0.0075
        self.ak39 = 0.00001
        self.ak40 = 0.00088

        self.PKCtheta = 2000.0
        self.Ptotal = 1000.0
        self.JNKtotal = 1000.0

        # Initialize JNK state variables
        self.x = np.zeros(23)
        self.x[8] = 5.05386E-02  # [C*]n (nM)
        self.x[9] = 9.14784E-03  # [C*]c (nM)
        self.x[10] = 4.91984E+01  # [C]n (nM)
        self.x[11] = 9.71085E+00  # [C]c (nM)
        self.x[18] = 0.0  # [JNK*] (%)
        self.x[19] = 0.0  # [c-Jun] (%)
        self.x[20] = 0.0  # [JNK:Cn] (%)
        self.x[21] = 0.0  # [JNK:PKC] (%)
        self.x[22] = 0.0  # [JNK:PKC:Cn] (%)

    def derivatives(self, cai):
        dx = np.zeros(23)
        
        # Update calcium concentrations
        self.x[12] = cai * 1000  # [Ca]n (nM)
        self.x[13] = cai * 1000  # [Ca]c (nM)

        dx[8] = (-self.ak15 * self.x[8] + self.ak16 * self.x[12]**3 * self.x[10] - 
                 self.ak13 * self.x[8] + self.ak14 * self.x[9])
        dx[9] = (-self.ak15 * self.x[9] + self.ak16 * self.x[13]**3 * self.x[11] + 
                 self.ak13 * self.x[8] - self.ak14 * self.x[9])
        dx[10] = (self.ak15 * self.x[8] - self.ak16 * self.x[10] * self.x[12]**3 + 
                  self.ak18 * self.x[11] - self.ak17 * self.x[10])
        dx[11] = (self.ak15 * self.x[9] - self.ak16 * self.x[11] * self.x[13]**3 - 
                  self.ak18 * self.x[11] + self.ak17 * self.x[10])
        dx[18] = 0.0
        dx[19] = 0.0
        dx[20] = (self.ak35 * (1 - self.x[20] - self.x[21] - self.x[22]) * self.x[9] - 
                  self.ak36 * self.x[20] + self.ak42 * self.x[22] - 
                  self.ak41 * self.x[20] * self.PKCtheta)
        dx[21] = (self.ak37 * (1 - self.x[20] - self.x[21] - self.x[22]) * self.PKCtheta - 
                  self.ak38 * self.x[21] + self.ak40 * self.x[22] - 
                  self.ak39 * self.x[21] * self.x[9])
        dx[22] = (self.ak39 * self.x[21] * self.x[9] - self.ak40 * self.x[22] + 
                  self.ak41 * self.x[20] * self.PKCtheta - self.ak42 * self.x[22])

        return dx

    def get_active_fraction(self, x):
        return x[22] / self.JNKtotal

# Main simulation function
def run_simulation(total_time=2400, dt=0.01):  # Increased dt for stability
    model = TCellModel()
    results = model.run_simulation(total_time, dt)
    model.plot_results(results)
    return results

# Run the simulation
if __name__ == "__main__":
    results = run_simulation()
    print("Simulation completed.")