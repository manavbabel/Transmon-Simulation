# set up a transmon
# note that 8 dimensions seems good to simulate the transmon without significant errors due to truncation

import numpy as np
from qutip import *
from copy import deepcopy
from scipy.optimize import fsolve
from math import isclose

class Transmon:

    def __init__(self, n_levels:int, initial_state=0, Ω:float=2*np.pi*5000, α:float=2*np.pi*-350, dt:float=1/20000, t_decay:float=np.inf, t_dephase:float=np.inf, RWA=False, A_noise=0, φ_noise=0):
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

        self.e_ops = [basis(n_levels, i) * basis(n_levels, i).dag() for i in range(n_levels)]

        self.X90_args = None

        a = destroy(n_levels)

        # note the Hamiltonians are divided by ħ already
        if RWA:
            self.H0 = Ω*a.dag()*a + 0.5*α*a.dag()*a*(a.dag()*a-1)
        else:
            self.H0 = self.make_H0()

        self.H1 = a + a.dag()

    def make_H0(self):

        a = destroy(self.n_levels)

        def H0_from_freqs(Ω,α):
            Ec = -α
            Ej = (Ω + Ec)**2 / (2*Ec)

            nhat = 1j * (Ej/(32*Ec))**0.25 * (a.dag()-a)
            φhat = (2*Ec/Ej)**0.25 * (a.dag()+a)

            H0 = 4*Ec*nhat**2 - Ej*φhat.cosm()
            return H0

        def find_actual_freqs(H0):
            eigs = H0.eigenenergies()
            Ω_true = np.real(eigs[1]-eigs[0])
            α_true = np.real((eigs[2]-eigs[1])-(eigs[1]-eigs[0]))
            return Ω_true,α_true

        def optimise_Ω(Ω_in,α_in):
            if isinstance(Ω_in, np.int32) or isinstance(Ω_in, np.float64):
                Ω2, _ = find_actual_freqs(H0_from_freqs(Ω_in,α_in))
                return Ω2 - self.Ω
            else:
                return [optimise_Ω(i,α_in) for i in Ω_in]
            
        def optimise_α(α_in,Ω_in):
            if isinstance(α_in, np.int32) or isinstance(α_in, np.float64):
                _, α2 = find_actual_freqs(H0_from_freqs(Ω_in,α_in))
                return α2 - self.α
            else:
                return [optimise_α(i,Ω_in) for i in α_in]
            
        Ω = fsolve(optimise_Ω, self.Ω, self.α)[0]
        α = fsolve(optimise_α, self.α, Ω)[0]
        for i in range(10):
            Ω = fsolve(optimise_Ω, Ω, α)[0]
            α = fsolve(optimise_α, α, Ω)[0]

        return H0_from_freqs(Ω,α)

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