# create X90 pulse

from copy import deepcopy
from scipy.interpolate import CubicHermiteSpline
from transmon_code.helpers import *
import numpy as np
from qutip import rand_ket
from transmon_code.simulate import *
import matplotlib.pyplot as plt

def create_X90_pulse(transmon, t=None, args=None, semiranges=None, plot=False, rand_init=False, N=20):
    # args is a dict of values for A,τg,Γ,λ,α,ω,φ with φ and offset optional and unnecessary
    # the mandatory ones are varied to find the optimum
    
    ψ0 = transmon.ψ0

    if args is None:
        args = {"A":15, "τ":0.24, "λ":0, "α":transmon.α, "ω":transmon.Ω}
    if semiranges is None:
        semiranges = {"A":10, "λ":10}

    if t is None:
        t = np.arange(0, args["τ"], transmon.dt)

    tmp_args = deepcopy(args)

    for parameter in ["A", "λ"]:

        if semiranges[parameter] == 0:
            print("Keeping "+parameter+" constant.")
            continue

        test_values = np.linspace(tmp_args[parameter]-semiranges[parameter], tmp_args[parameter]+semiranges[parameter], N)

        fidelities = []

        print("Optimising "+parameter)

        target = expand(calculate_target_state("X90", transmon.ψ0), transmon.n_levels).unit()

        for i, v in enumerate(test_values):
            tmp_args[parameter] = v
            if rand_init:
                tmp_fidelities = []
                for j in range(5):
                    transmon.ψ0 = expand(rand_ket(2), transmon.n_levels)
                    target = expand(calculate_target_state("X90",transmon.ψ0), transmon.n_levels).unit()
                    res = simulate(transmon, args=tmp_args, noise=False, plot=False)
                    tmp_fidelities.append(f)
                fidelities.append(np.mean(tmp_fidelities))

            else:
                res = simulate(transmon, args=tmp_args, noise=False, plot=False)
                # fidelities.append(f)
                if parameter == "λ":
                    fidelities.append(sum([expect(i, res[-1]) for i in transmon.e_ops[:2]]))
                    # fidelities.append(fidelity(res[-1], target)**2)
                elif parameter == "A":
                    fidelities.append(fidelity(truncate(res[-1]), truncate(target))**2)
                    # fidelities.append(fidelity(truncate(res[-1]), truncate(target))**2)
            
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
            else:
                plt.ylabel("Truncated fidelity")
            plt.title("Fidelity variation with " + parameter)
            plt.show()

    transmon.ψ0 = ψ0

    print("Optimal args:"+str(tmp_args))
    return tmp_args 