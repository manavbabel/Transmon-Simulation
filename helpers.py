# helper functions and variables

from qutip import *
from qutip_qip import circuit
import matplotlib.pyplot as plt 
import numpy as np
from IPython.display import display, Math

π = np.pi

# the tuples are (θ, φ, λ)

gate_angles = {
    "I": (0, 0, 0),
    "X180" : (π, 0, 0),
    "X" : (π, 0, 0),
    "Y180" : (π, π/2, -π/2),
    "Y" : (π, π/2, -π/2),
    "Z180" : (0, π/2, π/2),
    "Z" : (0, π/2, π/2),
    "X90" : (π/2, 0, 0),
    "SQRTX": (π/2, 0, 0),
    "Y90" : (π/2, π/2, -π/2),
    "SQRTY" : (π/2, π/2, -π/2),
    "Z90" : (0, π/4, π/4),
    "S" : (0, π/4, π/4),
    "X45" : (π/4, 0, 0),
    "Z45" : (0, π/8, π/8),
    "T" : (0, π/8, π/8),
    "H" : (π/2, π/2, π/2)
}

# function that returns the general U(θ, φ, λ) gate in the form of a 2x2 Qobj
# source: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.96.022330
def U(angles):
    θ, φ, λ = angles
    return Qobj(np.array(
        [
            [np.cos(θ/2), -1j*np.exp(1j*λ)*np.sin(θ/2)], 
            [-1j*np.exp(1j*φ)*np.sin(θ/2), np.exp(1j*(φ+λ))*np.cos(θ/2)]
            ]
        ))

# truncates states and density matrices to 2D
def truncate(arr:Qobj):

    try:
        # if arr is a state vector, truncates to first two elements
        if arr.dims[1][0] == 1:
            arr = Qobj(arr.full()[:2])
        # if arr is a square matrix, gets upper left 2x2 square
        else:
            arr = Qobj([i[:2] for i in arr.full()[:2]])
    except:
        raise ValueError("arr must be a vector or square matrix of dimension>=2.")

    return arr

def expand(arr:Qobj, D:int):
    try:
        if arr.dims[1][0] == 1:
            arr = Qobj(np.pad(arr.full(), ((0,D-len(arr.full())),(0,0))))
        else:
            arr = Qobj(np.pad(arr.full(), ((0,D-len(arr.full())),(0,D-len(arr.full())))))

    except:
        raise ValueError("arr must be a vector or square matrix of dimension=2.")

    return arr
    
# calculates the end result of a set of gates        
def calculate_target_state(circ, initial_state, plot=False):
    # circ is a gate name or list where elements are gate names, matrices, or Qobjs
    # initial state is a Qobj
    # if it is not 2D, truncate to 2D

    # reduces initial_state to 2D
    initial_state = truncate(initial_state)

    if not isinstance(circ, list):
        circ = [circ]

    # converts all gates to 2D Qobjs
    for i, v in enumerate(circ):
        if isinstance(v, str):
            try:
                circ[i]= U(gate_angles[v.upper()])
            except KeyError:
                raise KeyError("Gate not recognised.")
        elif not isinstance(v, Qobj):
            circ[i] = truncate(Qobj(v))
        else:
            circ[i] = truncate(circ[i])

    def custom_gate(ind):
        return circ[ind]

    qc = circuit.QubitCircuit(1)
    qc.user_gates = {"custom": custom_gate}
    [qc.add_gate("custom", targets=[0], arg_value=i) for i in range(len(circ))]

    final = qc.run(initial_state).unit()

    if plot:
        b = Bloch()
        b.make_sphere()
        b.add_states(initial_state)
        b.add_states(final)
        b.render()
        b.show()
        plt.show()

    return final

# decomposes a given gate into the U angles θ, φ, λ
def decompose_gate(gate:Qobj):
    # gate is a Qobj

    mat = gate.full()

    # remove the global phase
    global_phase = np.angle(mat[0][0])
    mat = mat * np.exp(-1j * global_phase)

    # calculate θ
    θs = []
    θs.append(2*np.arccos(mat[0][0]))
    θs.append(2*np.arccos(np.abs(mat[1][1])))
    θs.append(2*np.arcsin(np.abs(mat[0][1])))
    θs.append(2*np.arcsin(np.abs(mat[1][0])))
    θ = np.mean(θs)

    # if θ==0 then we can't derive the other angles from the 10 and 01 components
    # instead, we assume the angle is split evenly between the two, and halve the angle of the 11 component
    if θ == 0:
        φ = np.angle(mat[1][1])/2
        λ = np.angle(mat[1][1])/2
    else:
        φ = np.angle(mat[1][0])+π/2
        λ = np.angle(mat[0][1])+π/2

    return np.real([θ, φ, λ, global_phase])

def rotate_z(state, θ):
    # state is a ket or density matrix

    RZ = np.eye(state.dims[0][0], dtype=np.complex128)
    RZ[0][0] = np.exp(-1j*θ/2)
    RZ[1][1] = np.exp(1j*θ/2)
    RZ = Qobj(RZ)

    if state.isket:
        return RZ * state
    else:
        # need to use the superoperator formalism
        # see
        # * https://quantumcomputing.stackexchange.com/questions/23935/acting-with-a-superoperator-to-states-in-qutip
        return vector_to_operator(to_super(RZ)*operator_to_vector(state))
    
def clean(results):
    # results is a list of Qobjs
    # this function is necessary due to floating-point errors

    cleaned_results = []

    for result in results:
        # round to remove floating point errors
        result_tmp = np.round(result.full(), 5)

        # instead, uses inbuilt function
        result_tmp = result.tidyup()

        # make the diagonals all real
        if not result_tmp.isherm:
            for i in range(result.dims[0][0]):
                result_tmp[i][i] = np.real(result_tmp[i][i])

        if not result_tmp.isherm:
            raise ValueError("Received result is not hermitian", result)

        cleaned_results.append(Qobj(result_tmp).unit(inplace=False, norm="tr"))

    return cleaned_results

def plot_bloch(states):

    for i, v in enumerate(states):
        if v.isoper and not v.isherm:
            print("State in position "+str(i)+" of input is not Hermitian.")
            raise ValueError(v)

    b = Bloch()
    b.make_sphere()
    b.add_states(states)
    b.render()
    b.show()

def disp(qobj):
    # displays the matrix elements of a Qobj in formatted LaTeX
    display(Math("$"+qobj._repr_latex_().split("$")[1]+"$"))

def _E(a,b,n):
    ea, eb = np.zeros(n), np.zeros(n)
    ea[a], eb[b] = 1, 1
    return ea[:, np.newaxis] * eb

def generate_basis(n):
    # uses https://math.stackexchange.com/questions/91598/what-is-a-basis-for-the-space-of-n-times-n-hermitian-matrices
    P = []
    P.append([_E(i,i,n) for i in range(n)])
    P.append([[(_E(i,j,n)+_E(j,i,n))/np.sqrt(2) for j in range(i+1, n)] for i in range(n)])
    P.append([[1j*(_E(i,j,n)-_E(j,i,n))/np.sqrt(2) for j in range(i+1, n)] for i in range(n)])
    P = [i for j in P for i in j]

    P_clean = []
    for i in P:
        if isinstance(i, list):
            [P_clean.append(j) for j in i]
        else:
            P_clean.append(i)
    return P_clean
