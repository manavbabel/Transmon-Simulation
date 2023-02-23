# superoperator formalism helper functions

from transmon_code.helpers import *
from qutip import *
from transmon_code.simulate import *
from copy import deepcopy
from transmon_code.RBT import RBT_circuit

class PTMs:

    def __init__(self, dim):
        
        self.dim = dim
        self.basis = self.get_basis()

    # define basis
    def get_basis(self):
        # there are two options for this
        # 1. get the full basis
        # 2. get the 2D basis and expand, leaving 4 basis matrices 

        # first method
        basis = generate_basis(self.dim)

        # second method
        # basis = [expand(Qobj(i), self.dim) for i in generate_basis(2)]

        """# OR a d-level basis but reduced

        # basis_reduced = []
        # 
        # for i, m in enumerate(basis):
        #     tmp = [i[:2] for i in m.full()[:2]]
        # 
        #     if all([abs(i)==0 for j in tmp for i in j]):
        #         basis_reduced.append(Qobj(np.zeros(m.shape)))
        #     else:
        #         basis_reduced.append(m)
        # 
        # basis = basis_reduced"""

        return basis

    def to_vec(self, ρ:Qobj):
        # ρ is a ket or density matrix
        ρ = deepcopy(ρ)
        if ρ.isket:
            ρ = ket2dm(ρ)
        if ρ.dims[0][0] != self.dim:
            raise AttributeError("Dimension mismatch in to_vec: ρ is not of the right dimension.")

        return Qobj([[np.trace(i*ρ) for i in self.basis]]).trans() #to convert to column

    def to_oper(self, vec:Qobj):
        # vec is a vector from to_vec
        vec = deepcopy(vec)
        if vec.dims[1][0] != 1:
            raise AttributeError("Error in to_oper: vec is not a vector.")
        return sum([vec.full()[i][0] * self.basis[i] for i in range(vec.dims[0][0])])

    def apply_PTM(self, PTM, state):
        # state is a ket or density matrix
        # returns a density matrix

        state = deepcopy(state)
        PTM = deepcopy(PTM)

        if state.isket:
            state = ket2dm(state)

        return self.to_oper(PTM*self.to_vec(state))

    # calculate the ideal PTM for some gate
    def ideal_PTM(self, gate:str, Zθ=0):
        # gate is either a str listed in gate_angles or "Z"

        if gate == "Z":
            gate_op = expand(Qobj([[1,0],[0, np.exp(1j*Zθ)]]), self.dim)
        else:
            gate_op = expand(U(gate_angles[gate]), self.dim)

        transformed_basis = [gate_op*j*gate_op.dag() for j in self.basis]

        return Qobj([[np.trace(i.dag()*j) for j in transformed_basis] for i in self.basis])

    def learn_X90_PTM(self, transmon):

        transmon = deepcopy(transmon)

        if transmon.n_levels != self.basis[0].dims[0][0]:
            raise AttributeError("Dimension mismatch between transmon and basis.")
        
        ψ0 = transmon.ψ0

        def _run_pulse(init):
            transmon.ψ0 = init
            results = simulate(transmon, transmon.X90_args)
            return results[-1]
        
        transformed_basis = [_run_pulse(j) for j in self.basis]

        transmon.ψ0 = ψ0

        return Qobj([[np.trace(i.dag()*j) for j in transformed_basis] for i in self.basis])

    def circuit_PTM(self, circ, X90_PTM):
        # circ is a list of strings or Qobjs
        # returns a PTM representing the circuit
        PTM = Qobj(np.eye(self.dim**2))

        if not isinstance(circ, list):
            circ=[circ]

        for gate in circ:
            if isinstance(gate, str):
                gate = U(gate_angles[gate])
            θ, φ, λ, _ = decompose_gate(gate)

            PTM = self.ideal_PTM("Z", φ-np.pi/2) * X90_PTM * self.ideal_PTM("Z", np.pi-θ) * X90_PTM * self.ideal_PTM("Z", λ-np.pi/2) * PTM

        return PTM
    
    def RBT_PTM(self, transmon, X90_PTM, lengths, iterations):

        transmon.ψ0 = basis(transmon.n_levels, 0)

        fidelities = []

        for length in lengths:
            print("Testing length "+str(length))

            tmp_fidelities = []

            for iteration in range(iterations):
                print(iteration+1, end=" ")

                circ = RBT_circuit(length)

                res = self.apply_PTM(self.circuit_PTM(circ, X90_PTM), transmon.ψ0)

                tmp_fidelities.append(fidelity(transmon.ψ0, res))

            print("", end="\n")

            fidelities.append(np.mean(tmp_fidelities))

        print(fidelities)

        plt.plot(lengths, fidelities)
        plt.xticks(lengths)
        plt.xlabel("Circuit length")
        plt.ylabel("Mean fidelity")
        plt.show()

        return fidelities
