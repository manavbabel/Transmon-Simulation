# create X90 pulse

from copy import deepcopy
from scipy.interpolate import CubicHermiteSpline
from transmon_code.helpers import *
import numpy as np
from qutip import rand_ket
from transmon_code.simulate import *
import matplotlib.pyplot as plt

def create_X90_pulse(transmon, args, semiranges, plot=False, rand_init=False, N=40):
    # args is a dict of values for A,τ,λ with Ω,α,φ, and offset optional and unnecessary
    # A,λ are varied to find the optimum
    
    ψ0 = transmon.ψ0
    tmp_args = deepcopy(args)

    for parameter in ["A","λ","δ"]:

        if semiranges[parameter] == 0:
            print("Keeping "+parameter+" constant.")
            continue

        test_values = np.linspace(tmp_args[parameter]-semiranges[parameter], tmp_args[parameter]+semiranges[parameter], N)

        fidelities = []

        print("Optimising "+parameter)

        for i, v in enumerate(test_values):
            tmp_args[parameter] = v

            if rand_init:
                ψ0s = [expand(rand_ket(2), transmon.n_levels) for k in range(20)]
            else:
                ψ0s = [transmon.ψ0]

            targets = [expand(calculate_target_state("X90",i), transmon.n_levels).unit() for i in ψ0s]
            results = [simulate(setattr(transmon,"ψ0",i) or transmon, args=tmp_args)[-1] for i in ψ0s]

            # if fixed detuning, put δ here?
            if parameter=="A" or parameter=="δ":
                fidelities.append(np.mean([fidelity(truncate(res_i).unit(), truncate(targ_i).unit())**2 for res_i, targ_i in zip(results, targets)]))

            # if variable detuning, put δ here?
            elif parameter=="λ":
                fidelities.append(np.mean([sum([expect(i,res_i) for i in transmon.e_ops[:2]]) for res_i in results]))

            print(i+1, end=" ")

        print("", end="\n")

        fidelity_func = CubicHermiteSpline(test_values, fidelities, np.gradient(fidelities, test_values))

        cr_pts = [i for i in fidelity_func.derivative().roots() if i >= test_values[0] and i <= test_values[-1]]

        if len(cr_pts) == 0:
            tmp_args[parameter] = test_values[np.argmax(fidelities)]
            print("No optimum found for " + parameter + ", optimal value set to end of test values at " +str(test_values[-1]) + ". Rerun with altered semiranges.")
        else:
            tmp_args[parameter] = cr_pts[np.argmax(fidelity_func(cr_pts))]  

        if plot:
        
            plt.plot(test_values, fidelities)
            plt.axvline(tmp_args[parameter], c='r')
            plt.xlabel(parameter)
            if parameter == "λ":
                plt.ylabel("1 - leakage")
                plt.title("Leakage variation with λ")
            else:
                plt.ylabel("Truncated fidelity")
                plt.title("Fidelity variation with "+str(parameter))
            plt.show()

    transmon.ψ0 = ψ0

    print("Optimal args:"+str(tmp_args))
    return tmp_args