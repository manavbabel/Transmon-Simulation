# optimiser class

import numpy as np
from qutip import *
from qutip_qip import operations, circuit
from copy import deepcopy
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt

def calculate_target_state(initial_state, gates):
    # initial_state is a Qobj representing a state
    # gate is a custom Gate object

    if not isinstance(gates, list):
        gates = [gates]

    # truncate to 2D, keeping only the first two dimensions
    if initial_state.dims[0][0] > 2:
            initial_state = Qobj(initial_state.full()[:-1])

    initial_state = initial_state.unit()

    qc = circuit.QubitCircuit(1)

    for g in gates:
        qc.add_gate(operations.Gate(g.name, targets=0, arg_value=g.angle*2*np.pi/360))
    return qc.run(initial_state)

class Optimiser:

    def __init__(self, t, transmon):
        self.t = t
        self.transmon = transmon

    def optimise_parameter(self, gate, parameter, semirange, target_state, N=20, plot=False):

        tmp_args = deepcopy(gate.optimal_parameters)

        H = [self.transmon.H0, [self.transmon.H1, gate.H1_coeffs]]

        test_values = np.linspace(tmp_args[parameter]-semirange, tmp_args[parameter]+semirange, N)

        fidelities = []

        for i in test_values:
            tmp_args[parameter] = i
            results = mesolve(H, self.transmon.ψ0, self.t, self.transmon.c_ops, args=tmp_args)
            fidelities.append(fidelity(results.states[-1].unit(), target_state.unit()))

        fidelity_func = CubicHermiteSpline(test_values, fidelities, np.gradient(fidelities, test_values))

        cr_pts = [i for i in fidelity_func.derivative().roots() if i >= test_values[0] and i <= test_values[-1]]

        if len(cr_pts) == 0:
            tmp_args[parameter] = test_values[np.argmax(fidelities)]
            print("No optimum found, optimal value set to upper end of test values. Rerun with altered parameters.")

        else:
            tmp_args[parameter] = cr_pts[np.argmax(fidelity_func(cr_pts))]  

        if plot:
        
            plt.plot(test_values, fidelities)
            plt.axvline(tmp_args[parameter], c='r')
            plt.xlabel(parameter)
            plt.ylabel("Fidelity")
            plt.title("Fidelity variation with " + parameter)
            plt.show()

        return tmp_args
    
    def optimise_gate(self, gate, semiranges, initial_args=None, plot=False):
        # NOTE: initial_args will only be used if the gate has no predefined optimal parameters

        if not gate.optimal_parameters:
            gate.optimal_parameters = initial_args

        target_state = Qobj(np.append(calculate_target_state(self.transmon.ψ0, gate).full(), 0))

        for param in ["A", "ω", "Γ", "A_DRAG"]:
            gate.optimal_parameters = self.optimise_parameter(gate, param, semiranges[param], target_state, plot=plot)

    def optimise_gateset(self, gateset, gateset_semiranges, plot=False):

        for g in gateset:
            self.optimise_gate(g, semiranges = gateset_semiranges[g], plot=plot)

        return gateset