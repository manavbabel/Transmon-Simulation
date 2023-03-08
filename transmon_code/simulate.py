# simulate a pulse

from qutip import *
from transmon_code.shapes import *
from transmon_code.helpers import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def simulate(transmon, args, target=None, noise=False, plot=False):

    args = deepcopy(args)
    t = np.arange(0, args["τ"], transmon.dt)
    H = [transmon.H0, [transmon.H1, H1_coeffs]]

    # noise hasn't been touched in a while, go over it
    if noise:
        if transmon.A_noise == 0 or transmon.φ_noise == 0:
            print("Warning: either A_noise or φ_noise is zero for this transmon.")

        args["A"] += np.random.normal(0, transmon.A_noise)

        if "φ" in args.keys():
            args["φ"] += np.random.normal(0, transmon.φ_noise)
        else:
            args.update({"φ": np.random.normal(0, transmon.φ_noise)})

    if transmon.t_decay==np.inf and transmon.t_dephase==np.inf:
        results = mesolve(H, transmon.ψ0, t, args=args, options=Options(atol=1e-15, nsteps=10000, tidy=True))
    else:
        a = destroy(transmon.n_levels)
        c_ops = [np.sqrt(1/transmon.t_decay)*a, np.sqrt(1/transmon.t_dephase)*(a.dag()*a)]
        results = mesolve(H, transmon.ψ0, t, c_ops=c_ops, args=args, options=Options(atol=1e-15, nsteps=10000, tidy=True))

    results_rot = [rotate_z(s, transmon.Ω*t_i) for s, t_i in zip(results.states, t)]
    
    if plot:

        t_tmp = np.arange(0,args["τ"], 1/100000)

        plt.plot(t_tmp, H1_coeffs(t_tmp, args))
        plt.show()

        if target:
            break_down_errors(transmon, args["τ"], results_rot[-1], fidelity(results_rot[-1], target)**2 )
            plot_bloch(results_rot[::int(len(results_rot)/50)]+[target])
        else:
            plot_bloch(results_rot[::int(len(results_rot)/50)])

    if target:
        if fidelity(results_rot[-1], target)**2 >= 1:
            print(fidelity(results_rot[-1], target)**2)
            if results_rot[-1].isket:
                print(results_rot[-1].norm())
            else:
                print(results_rot[-1].norm("fro"))
            raise RuntimeError("Fidelity is measured to be >= 1.")
        return results_rot, fidelity(results_rot[-1], target)**2
    else:
        return results_rot
    
def simulate_circuit2(transmon, circuit, plot=True):
    # circuit is a Qobj or list of them

    if not isinstance(circuit, list):
        circuit = circuit

    target = expand(calculate_target_state(circuit, transmon.ψ0), transmon.n_levels).unit()

    state = deepcopy(transmon.ψ0)
    H = [transmon.H0, [transmon.H1, H1_coeffs]]

    args = deepcopy(transmon.X90_args)
    args.update({"offset":0})

    t = np.arange(0, args["τ"], transmon.dt)

    a = destroy(transmon.n_levels)
    c_ops = [np.sqrt(1/transmon.t_decay)*a, np.sqrt(1/transmon.t_dephase)*(a.dag()*a)]

    for gate in circuit:
        θ, φ, λ, _ = decompose_gate(gate)

        state = rotate_z(state, λ - np.pi/2)

        res = mesolve(H, state, t, c_ops=c_ops, args=args, options=Options(atol=1e-15, nsteps=10000, tidy=True))
        state = rotate_z(res.states[-1], transmon.Ω*t[-1]+args["final_Z_rot"])
        
        state = rotate_z(state, np.pi - θ)

        args["offset"] += args["τ"]
        res = mesolve(H, state, t, c_ops=c_ops, args=args, options=Options(atol=1e-15, nsteps=10000, tidy=True))
        state = rotate_z(res.states[-1], transmon.Ω*t[-1]+args["final_Z_rot"])

        state = rotate_z(state, φ - np.pi/2)

        args["offset"] += args["τ"]

    fid = fidelity(state, expand(target, transmon.n_levels).unit())**2

    if plot:
        break_down_errors(transmon, 2*len(circuit)*transmon.X90_args["τ"], state, fid)
        plot_bloch([transmon.ψ0, state, target])

    return state, fid

def simulate_circuit(transmon, circuit, noise=False, plot=True):
    # circuit is a Qobj or a list of them

    if not isinstance(circuit, list):
        circuit=[circuit]
    
    t = np.arange(0, 2*len(circuit)*transmon.X90_args["τ"], transmon.dt)

    target = expand(calculate_target_state(circuit, transmon.ψ0), transmon.n_levels).unit()

    angles = [decompose_gate(i) for i in circuit]
    θs, φs, λs = [[i[j] for i in angles] for j in range(3)]

    logical_gate_ends = np.cumsum([transmon.X90_args["τ"]*2]*len(circuit))

    if noise:
        raise NotImplementedError("Transmon noise has not been updated yet.")
        pulse_args = [transmon.get_noisy_args() for i in range(2*len(circuit))]
    else:
        pulse_args = [deepcopy(transmon.X90_args) for i in range(2*len(circuit))]

    def H1_coeffs_partial(t, args):
    # args is a dummy dict

        if not isinstance(t, float):
            return [H1_coeffs_partial(t_i, args) for t_i in t]
        else:
            logical_gate_number = np.clip(np.digitize(t, logical_gate_ends), 0, len(θs)-1)

            offset = 2*transmon.X90_args["τ"]*logical_gate_number
            φ = λs[logical_gate_number] - np.pi/2

            if t-offset >= transmon.X90_args["τ"]:
                offset += transmon.X90_args["τ"]
                φ += (np.pi - θs[logical_gate_number])
                tmp_args = deepcopy(pulse_args[2*logical_gate_number+1])
            else:
                tmp_args = deepcopy(pulse_args[2*logical_gate_number])

            if logical_gate_number != 0:
                φ += sum([λs[i]+φs[i]-θs[i] for i in range(logical_gate_number)])

            tmp_args.update({"φ":φ})
            tmp_args.update({"offset":offset})

            return H1_coeffs(t, tmp_args)
        
    H = [transmon.H0, [transmon.H1, H1_coeffs_partial]]

    a = destroy(transmon.n_levels)

    if transmon.t_decay==np.inf and transmon.t_dephase==np.inf:
        results = mesolve(H, transmon.ψ0, t, args={}, options=Options(atol=1e-15, nsteps=10000, tidy=True))
    else:
        a = destroy(transmon.n_levels)
        c_ops = [np.sqrt(1/transmon.t_decay)*a, np.sqrt(1/transmon.t_dephase)*(a.dag()*a)]
        results = mesolve(H, transmon.ψ0, t, c_ops=c_ops, args={}, options=Options(atol=1e-15, nsteps=10000, tidy=True))

    # results.states = [i.unit(norm="fro", inplace=False) if i.isoper and i.norm("fro")>1 else i.unit() if i.isket and i.norm()>1 else i for i in results.states]

    results_time_rotated = [rotate_z(ri, transmon.Ω*ti) for ri,ti in zip(results.states, t)]
    # results_time_rotated = make_hermitian(results_time_rotated)

    total_φ = sum(φs) + sum(λs) - sum(θs)
    res = rotate_z(results_time_rotated[-1], total_φ)

    fid = fidelity(res, expand(target, transmon.n_levels).unit())**2

    if plot:

        break_down_errors(transmon, 2*len(circuit)*transmon.X90_args["τ"], res, fid)

        plt.plot(t, H1_coeffs_partial(t, {}))
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

        # plt.bar(range(transmon.n_levels), [np.real(expect(i, res)) for i in transmon.# e_ops])
        # plt.legend(["n="+str(i) for i in range(transmon.n_levels)])
        # plt.plot()

        plot_bloch(results_time_rotated[::20]+[res, target])
        
    return res, fid