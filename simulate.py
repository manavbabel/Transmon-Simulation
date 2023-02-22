# simulate a pulse

from qutip import *
from shapes import *
from helpers import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def simulate(transmon, args, t=None, target=None, noise=False, plot=False):

    args = deepcopy(args)

    if t is None:
        t = np.arange(0, args["Γ"], transmon.dt)

    H = [transmon.H0, [transmon.H1, H1_coeffs]]

    if noise:
        if transmon.A_noise == 0 or transmon.φ_noise == 0:
            print("Warning: either A_noise or φ_noise is zero for this transmon.")

        args["A"] += np.random.normal(0, transmon.A_noise)

        if "φ" in args.keys():
            args["φ"] += np.random.normal(0, transmon.φ_noise)
        else:
            args.update({"φ": np.random.normal(0, transmon.φ_noise)})

    results = mesolve(H, transmon.ψ0, t, c_ops=transmon.c_ops, args=args)

    try:
        # results = clean(results.states)
        results = results.states
        results_rot = [rotate_z(s, transmon.Ω*t_i) for s, t_i in zip(results, t)]
    except:
        print(results)
        raise RuntimeError("Error in cleaning or rotating results.")
    
    if plot:

        plt.plot(t, H1_coeffs(t, args))
        plt.show()

        res = [truncate(i) for i in results_rot]

        if target:
            plot_bloch(clean(res[::20])+[truncate(target)])
        else:
            plot_bloch(clean(res[::20]))

    if target:
        # if fidelity(results_rot[-1], target) > 1:
        #     print(results[-1].norm())
        #     print()
        #     print(results_rot[-1].norm())
        #     print()
        #     raise RuntimeError()
        return results_rot, fidelity(results_rot[-1], target)
    else:
        return results_rot
    
def simulate_circuit(transmon, circuit, noise=False, plot=True):
    # circuit is a Qobj or a list of them

    if not isinstance(circuit, list):
        circuit=[circuit]
    
    t = np.arange(0, 2*transmon.X90_args["Γ"]*len(circuit), transmon.dt)

    target = expand(calculate_target_state(circuit, transmon.ψ0), transmon.n_levels).unit()

    angles = [decompose_gate(i) for i in circuit]
    θs = [i[0] for i in angles]
    φs = [i[1] for i in angles]
    λs = [i[2] for i in angles]

    logical_gate_ends = np.cumsum([transmon.X90_args["Γ"]*2]*len(circuit))

    if noise:
        pulse_args = [transmon.get_noisy_args() for i in range(2*len(circuit))]
    else:
        pulse_args = [deepcopy(transmon.X90_args) for i in range(2*len(circuit))]
        [i.update({"φ":0}) for i in pulse_args]

    def H1_coeffs_partial(t, args):
    # args is a dummy dict

        if not isinstance(t, float):
            return [H1_coeffs_partial(t_i, args) for t_i in t]
        else:
            logical_gate_number = np.clip(np.digitize(t, logical_gate_ends), 0, len(θs)-1)

            offset = 2*transmon.X90_args["Γ"]*logical_gate_number
            φ = λs[logical_gate_number] - np.pi/2

            if t-offset >= transmon.X90_args["Γ"]:
                offset += transmon.X90_args["Γ"]
                φ += (np.pi - θs[logical_gate_number])
                tmp_args = deepcopy(pulse_args[logical_gate_number+1])
            else:
                tmp_args = deepcopy(pulse_args[logical_gate_number])

            if logical_gate_number != 0:
                φ += sum([λs[i]+φs[i]-θs[i] for i in range(logical_gate_number)])

            tmp_args["φ"] += φ
            tmp_args.update({"offset":offset})

            return H1_coeffs(t, tmp_args)
        
    H = [transmon.H0, [transmon.H1, H1_coeffs_partial]]
    results = mesolve(H, transmon.ψ0, t, c_ops=transmon.c_ops, args={})
    results_time_rotated = [rotate_z(i, transmon.Ω*t_i) for i, t_i in zip(results.states, t)]
    # results_time_rotated = clean(results_time_rotated)
    total_φ = sum(φs) + sum(λs) - sum(θs)
    res = rotate_z(results_time_rotated[-1], total_φ)
    f = fidelity(res, expand(target, transmon.n_levels).unit())

    if plot:

        print("Fidelity: " + str(f))
        print("Leakage error: " + str(sum([np.abs(expect(i, res)) for i in transmon.e_ops[2:]])))

        plt.plot(t, H1_coeffs_partial(t, None))
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

        plt.bar(range(transmon.n_levels), [np.real(expect(i, res)) for i in transmon.e_ops])
        plt.legend(["n="+str(i) for i in range(transmon.n_levels)])
        plt.plot()

        plot_bloch([truncate(transmon.ψ0), truncate(res)])
        
    return res, f