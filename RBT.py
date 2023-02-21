# randomised benchmarking module

import numpy as np
from simulator import simulate_circuit
import matplotlib.pyplot as plt

def RBT(gateset, transmon, N_tests=50, max_seq_length=10, plot=False, dt=1/1000,):
    # max_seq_length is the max mumber of *forward* gates to include.
    # gateset must be in the form [(X, X_inverse), ...] etc

    performance = []
    gates = np.array([i[0] for i in gateset])
    inverses = np.array([i[1] for i in gateset])

    for seq_length in range(1, max_seq_length+1):

        print("", end="\n")
        print("Testing sequences of length "+str(seq_length), end="\n")
        print("o"*N_tests)

        fidelities = []

        for _ in range(N_tests):

            test_gates = []

            gate_nums = np.random.randint(0, len(gateset), seq_length)
            test_gates.append(list(gates[gate_nums])[0])
            test_gates.append(list(inverses[gate_nums][::-1])[0])
            # print("Test gates: " + ", ".join([i.axis+str(i.angle) for i in test_gates]))
            _, final_fidelity = simulate_circuit(test_gates, transmon, plot=False, dt=dt)

            fidelities.append(final_fidelity)
            print("x", end="")

        performance.append(np.mean(fidelities))

    plt.plot(range(1, max_seq_length+1), performance)
    plt.show()

    return performance
