# RBT gate sequence function
# you need to implement collapse operators!!!!

from random import choice as random_choice
from qutip_qip.operations import qubit_clifford_group
from qutip import Qobj
import numpy as np
from helpers import *
import matplotlib.pyplot as plt
from simulate import *

clifford_1q = [i for i in qubit_clifford_group(1)]

def RBT_circuit(length):
    # length is the number of normal gates
    # returns a list of Qobjs of length length+1 where the final element is the inverse of all the previous elements
    
    circ = [random_choice(clifford_1q) for i in range(length)]

    tmp = Qobj(np.eye(2))

    for gate in circ[::-1]:
        tmp = tmp*gate

    circ.append(tmp.inv())
    return circ

def perform_RBT(transmon, max_length:int, iterations:int, noise=True, plot=True):
    
    if transmon.ψ0 != basis(transmon.n_levels, 0):
        raise RuntimeWarning("Warning: initial state of RBT testing is not 0.")

    fidelities = []

    for length in range(1, max_length+1):
        print("Testing length "+str(length))

        tmp_fidelities = [0]*iterations

        for iteration in range(iterations):
            print(iteration+1, end=" ")

            circ = RBT_circuit(length)

            _, f = simulate_circuit(transmon, circ, noise=noise, plot=False)
            tmp_fidelities[iteration] = f

        print("", end="\n")

        fidelities.append(np.mean(tmp_fidelities))

    print(fidelities)

    if plot:
        plt.plot(range(1, max_length+1), fidelities)
        plt.xticks(range(1, max_length+1))
        plt.xlabel("Circuit length")
        plt.ylabel("Mean fidelity")
        plt.show()

    return fidelities