# set up a transmon

import numpy as np
from qutip import *

class Transmon:

    def __init__(self, n_levels:int, initial_level:int, Ω:float=5000, α:float=-350, t_decay:float=np.inf, t_dephase:float=np.inf):
        
        self.n_levels = n_levels
        self.Ω = Ω
        self.α = α
        self.t_decay = t_decay
        self.t_dephase = t_dephase

        self.ψ0 = basis(n_levels, initial_level)

        a = destroy(n_levels)
        n = a.dag() * a

        self.e_ops = [basis(n_levels, i) * basis(n_levels, i).dag() for i in range(n_levels)]
        self.c_ops = [np.sqrt(1/t_decay) * a, np.sqrt(1/t_dephase) * a.dag()*a]

        Ec = -α
        Ej = (-1*(Ω-α)**2)/(8*α)

        # with RWA
        self.H0 = Ω*n + 0.5*α*n*(n-1)    
        
        # without RWA
        # decrease timesteps for more accurate analysis
        self.H0 = -1 * np.sqrt(Ej*Ec/2)*(a-a.dag())**2 - Ej * Qobj(np.cos(((2*Ec)/Ej)**0.25 * (a+a.dag()).full()))

        self.H1 = a + a.dag()