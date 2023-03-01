# Code Outline

NOTE: qutip's `fidelity` function needs to be squared to return the true fidelity!

## To-do

- see how truncation affects fidelity. let the n=8 parameters be the true ones, and see how final fidelity varies with no of levels
- try and quantify error budget from the [paper](https://www.science.org/doi/10.1126/sciadv.abl6698)
  1. for each qubit, optimise the X90 pulse to at least 5 nines fidelity
  2. make a list of args with noise added to A,φ, and train X90_PTMs for each one
  3. simulate the gates being applied to each paper, so that decoherence+leakage is roughly the value given in the rightmost column
  4. modify A,φ noise to account for the rest of the discrepancy (next two columns)
  5. use master's [paper](//C:/Users/manav/OneDrive/Oxford/Master's%20project/Papers/CMP1902-1qerrors.pdf) from two years ago to compare with noise in actual system
- first, need to check that the PTMs exactly replicate the pulse
- find z error after each pulse and add a term correcting for it? so the only pulse error is in θ, not φ
- optimise A for 01 fidelity, and then A_drag for leakage
- optimise frequency
- check units - does Γ correspond to width? what width is used in the paper?

## Aims

Replicate results in online paper from both pulse-level simulation and PTM, and quantify error budget - how much electronics noise is required to get those results?

also, to what level does the transmon need to be simulated to get some fidelity?

## Algorithm

1. Define a transmon class, which acts as a wrapper for various properties. Only the RWA Hamiltonian is currently supported - the rotating frame transformation for the non-RWA Hamiltonian has not been derived. Random noise is added to the optimal parameters upon calling the proper tunction.
2. Optimise an X90 pulse for this transmon, using randomised initial states. 
3. Run a custom circuit to check the functionality
4. Create a PTM
5. Check pulse level simulation and PTM simulation give same results
6. Add noise and compare to paper

## Further notes

Here, we decompose each gate in a circuit into the vZX basis, which has only an X90 gate, and virtual Z gates. This means the only physical gate we have to implement and optimise is the X90 gate, substantially improving computational time.

This decomposition is predicated on the fact that the general gate

$$U(\theta,\phi,\lambda)=\begin{pmatrix}
\cos{\theta/2}&-e^{-\imath\lambda}\sin{\theta/2}\\
e^{\imath\phi}\sin{\theta/2}&e^{\imath(\phi+\lambda)}\cos{\theta/2}
\end{pmatrix}$$

Can be deconstructed into

$$Z(φ-π/2) @ X(π/2) @ Z(π-θ) @ X(π/2) @ Z(λ-π/2)$$

> $@$ means matrix multiplication. It is important to note that numpy operations by default are elementwise!

Where the virtual $Z$ gates are implemented by shifting the phase of **all** the subsequent $X$ pulses.

Furthermore, it is easier to work in the rotating basis. Instead of applying the rotating frame transformation to the operators, which increases computational load to the point where the simulation is unusable, I have opted to simply apply a time-varying rotation to each state in the final result, without loss of generality.

However, the *frequency* of this rotation is currently only known for the transmon Hamiltonian which has been derived using the RWA. To simulate the full non-approximated transmon, the frequency at which the transmon oscillates under that Hamiltonian must be derived. It is then a matter of checking whether the RWA is being used, and using the appropriate frequency.

When doing spectroscopy, a spike in P(f) is seen at Ω02/2, meaning multiple-photon transitions are already included.
