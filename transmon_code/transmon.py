# set up a transmon
# note that 8 dimensions seems good to simulate the transmon without significant errors due to truncation

import numpy as np
from qutip import *
from copy import deepcopy
from scipy.optimize import fsolve

class Transmon:

    def __init__(self, n_levels:int, initial_state=0, Ω:float=5000, α:float=-350, dt:float=1/10000, t_decay:float=np.inf, t_dephase:float=np.inf, RWA=True, A_noise=0, φ_noise=0):
        # Ω and α are in radians per second
        # all times are in μs
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

        if RWA:
            self.H0 = Ω*a.dag()*a + 0.5*α*a.dag()*a*(a.dag()*a-1)
        else:

            Ω_actual, α_actual = self.calculate_actual()

            Ec = -1*α_actual
            Ej = -1 * (Ω_actual-α_actual)**2 / (8*α_actual)

            nzpf = (Ej/(32*Ec)) ** 0.25
            φzpf = ((2*Ec)/Ej) ** 0.25

            n = 1j * nzpf * (a.dag() - a)
            φ = φzpf * (a.dag() + a)

            self.H0 = 4*Ec*n**2 - Ej*φ.cosm()

        # note the Hamiltonians are divided by ħ already

        # self.Ω = np.real(self.H0.eigenenergies()[1]-self.H0.eigenenergies()[0])
        # if n_levels >2:
            # self.α = (self.H0.eigenenergies()[2]-self.H0.eigenenergies()[1])-(self.H0.eigenenergies()[1]-self.H0.eigenenergies()[0])

        self.H1 = a + a.dag()

        self.X90_args = None
    
    def calculate_actual(self):

        a = destroy(self.n_levels)

        def _calc(f, an):
            Ec = -1*an
            Ej = -1 * (f-an)**2 / (8*an)

            nzpf = (Ej/(32*Ec)) ** 0.25
            φzpf = ((2*Ec)/Ej) ** 0.25

            n = 1j * nzpf * (a.dag() - a)
            φ = φzpf * (a.dag() + a)

            H0 = 4*Ec*n**2 - Ej*φ.cosm()
            eigs = np.real(H0.eigenenergies())

            f2 = np.real(eigs[1]-eigs[0])
            an2 = (eigs[2]-eigs[1])-(eigs[1]-eigs[0])

            return f2, an2
        
        def optimise_Ω(f,an):
            if isinstance(f, np.int32) or isinstance(f, np.float64):
                f2, _ = _calc(f,an)
                return f2 - self.Ω
            else:
                return [optimise_Ω(i,an) for i in f]

        def optimise_α(an,f):
            if isinstance(an, np.int32) or isinstance(an, np.float64):
                _, an2 = _calc(f,an)
                return an2 - self.α
            else:
                return [optimise_α(i,f) for i in an]
            
        optimised_Ω, optimised_α = deepcopy(self.Ω), deepcopy(self.α)
        
        for i in range(20):
            optimised_Ω = fsolve(optimise_Ω, optimised_Ω, args=optimised_α)[0]
            optimised_α = fsolve(optimise_α, optimised_α, args=optimised_Ω)[0]

        return optimised_Ω, optimised_α

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
