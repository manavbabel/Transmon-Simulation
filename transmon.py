# set up a transmon
# note that 8 dimensions seems good to simulate the transmon without significant errors due to truncation

import numpy as np
from qutip import *
from copy import deepcopy

class Transmon:

    def __init__(self, n_levels:int, initial_state, Ω:float=5000, α:float=-350, dt:float=1/1000, t_decay:float=np.inf, t_dephase:float=np.inf, RWA=True, A_noise=0, φ_noise=0):
        # Ω and α are in MHz
        # all times are in microseconds (OR NANOSECONDS?)
        # initial_state is an int or a Qobj
        # A_noise and φ_noise are the stdevs of the normal from which the noise is drawn

        if isinstance(initial_state, int):
            initial_state = basis(n_levels, initial_state)
        elif not isinstance(initial_state, Qobj):
            raise ValueError("initial_state must be an int or a Qobj")
        
        self.n_levels = n_levels
        self.Ω = Ω
        self.α = α
        self.dt = dt
        self.t_decay = t_decay
        self.t_dephase = t_dephase
        self.RWA = RWA
        self.A_noise = A_noise
        self.φ_noise = φ_noise

        self.ψ0 = initial_state

        a = destroy(n_levels)

        self.e_ops = [basis(n_levels, i) * basis(n_levels, i).dag() for i in range(n_levels)]
        self.c_ops = [np.sqrt(1/t_decay) * a, np.sqrt(1/t_dephase) *(a.dag()*a)]

        if RWA:
            self.H0 = Ω*a.dag()*a + 0.5*α*a.dag()*a*(a.dag()*a-1)
        else:
            Ec = -1*α
            Ej = -1 * (Ω-α)**2 / (8*α)

            nzpf = (Ej/(32*Ec)) ** 0.25
            φzpf = ((2*Ec)/Ej) ** 0.25

            n = 1j * nzpf * (a.dag() - a)
            φ = φzpf * (a.dag() + a)

            self.H0 = 4*Ec*n**2 - Ej*φ.cosm()

        # redefine Ω and α from the eigenenergies
        # note the Hamiltonians are divided by ħ already

        self.Ω = self.H0.eigenenergies()[1]-self.H0.eigenenergies()[0]
        self.α = (self.H0.eigenenergies()[2]-self.H0.eigenenergies()[1])-(self.H0.eigenenergies()[1]-self.H0.eigenenergies()[0])

        self.H1 = a + a.dag()

        self.X90_args = None

    def get_noisy_args(self):

        try:
            _ = self.X90_args.keys()
        except:
            raise AttributeError("transmon.X90_args has not yet been defined.")

        tmp_args = deepcopy(self.X90_args)

        A = tmp_args["A"] + np.random.normal(0, self.A_noise)
        φ = np.random.normal(0, self.φ_noise)

        if "φ" in tmp_args.keys():
            φ += tmp_args["φ"]
            
        tmp_args.update({"A":A, "φ":φ})
        
        return tmp_args
