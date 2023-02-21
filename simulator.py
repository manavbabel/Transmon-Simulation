# simulation

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from optimiser import calculate_target_state
from copy import copy, deepcopy

def simulate_gate(t, gate, transmon):

    if "ω" not in gate.optimal_parameters.keys():
        raise AttributeError("The gate has not been assigned a frequency.")

    H = [transmon.H0, [transmon.H1, gate.H1_coeffs]]

    result = mesolve(H, transmon.ψ0, t, transmon.c_ops, transmon.e_ops, gate.optimal_parameters, options=Options(store_states=True))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(t, gate.H1_coeffs(t, gate.optimal_parameters))
    ax1.set_ylabel("Amplitude")

    [ax2.plot(t, i) for i in result.expect]
    ax2.legend(["n="+str(i) for i in range(3)])
    ax2.set_ylabel("Probability")
    plt.xlabel("Time [what units?]")
    plt.show()

    target_state = Qobj(np.append(calculate_target_state(transmon.ψ0, gate).full(), 0))

    print("Final fidelity: " + str(fidelity(result.states[-1].unit(), target_state.unit())))

    return result

def simulate_gateset(t, gateset, transmon):
    results = []
    for gate in gateset:
        results.append(simulate_gate(t, gate, transmon))
    return results

def simulate_circuit(gates_tmp, transmon, plot=False, dt=1/1000):

    gates = deepcopy(gates_tmp)

    widths = [gate.optimal_parameters["Γ"] for gate in gates]

    t_list = np.arange(0, sum(widths), dt)

    def H1_coeffs_partial(t, args):
        # args is a dummy dict!

        gate_number = np.digitize(t,np.cumsum(widths))
        gate_number = np.clip(gate_number, 0, len(gates)-1)

        if isinstance(t, float):
            # implying gate_number is an int            

            args = copy(gates[gate_number].optimal_parameters)

            if gate_number != 0:
                args.update({"offset":np.cumsum(widths)[gate_number-1]})

            coeff = gates[gate_number].H1_coeffs(t, args)

            return coeff

        else:
            coeffs = []
            for t_i, gate in zip(t, gate_number):

                args = copy(gates[gate].optimal_parameters)

                if gate != 0:
                    args.update({"offset":np.cumsum(widths)[gate-1]})
                coeffs.append(gates[gate].H1_coeffs(t_i, args))

            return coeffs
    
    H = [transmon.H0, [transmon.H1, H1_coeffs_partial]]

    result = mesolve(H, transmon.ψ0, t_list, transmon.c_ops, transmon.e_ops, args={}, options=Options(store_states=True))

    target_state = Qobj(np.append(calculate_target_state(transmon.ψ0, gates).full(), 0))
    final_fidelity = fidelity(result.states[-1].unit(), target_state.unit())

    if plot:

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(t_list, H1_coeffs_partial(t_list, {}))
        ax1.set_ylabel("Amplitude")

        [ax2.plot(t_list, i) for i in result.expect]
        ax2.legend(["n="+str(i) for i in range(3)])
        ax2.set_ylabel("Probability")
        plt.xlabel("Time [what units?]")

        for i in np.cumsum(widths):
            ax1.axvline((i))
            ax2.axvline((i))

        plt.show()

        print("Final fidelity: " + str(final_fidelity))

    return result, final_fidelity

