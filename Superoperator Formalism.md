# Superoperator Formalism

Here, I present a way to represent the effect of a pulse as a matrix, massively speeding up computation, and allowing more detailed error analysis.

We start with a recounting of the basics, considering a single $n$-level qubit. A pure state (ket) of this qubit $\ket{\psi}$ exists in an $n$-dimensional Hilbert space, and is represented by a vector of length $n$. An operator is a map from $\mathcal{H}_n\to\mathcal{H}_n$, and is a map from kets to other kets.

If we now include mixed states, allowing for non-unitary evolution, we must change formalism. Now, a state is represented by a *density operator* $\rho=\ket{\psi}\bra{\psi}$, which is represented as an $n\times n$ matrix. This is why they are also known as *density matrices*.

> It can be seen that some of the operators in the pure formalism can be seen as density matrices representing states. For example, the $X90$ operator corresponds to the state $\ket{0}+\ket{1}$.

The question is then how do we operate on these density matrices? A naïve way would be to simply change the way they operate:

$$A\ket{\psi}\to A\rho A^\dag$$

However, this does not offer much insight into the process. A better way is to use a representation known as a Pauli Transfer Matrix.

## PTMs

> Go over this - in particular, why have the dimensions changed, and why is it a 4x4 matrix?

Consider a linear map $\rho \to \Lambda(\rho)$, acting on a Hilbert space which has some basis $P$. In 2D, this can be the Pauli set. For higher dimensions, we use the `generate_basis` function from `helpers.py`.

Then, we have a change of dimension: density matrices, which were $n\times n$ matrices, become vectors of length $n^2$ on a Hilbert space $\mathcal{H}_{n^2}$, and are written as superkets $\ket{\rho}\rangle$. Operators become superoperators, and are represented by matrices of dimension $n^2\times n^2$.

Under this formalism, we can derive something called the Pauli Transfer Matrix:

$$\left(R_\Lambda\right)_{ij}=\frac{1}{n}\text{Tr }\left[P_i\Lambda\left(P_j\right)\right]$$

And we find that

$$\ket{\Lambda(\rho)}\rangle=R_\Lambda \ket{\rho}\rangle$$

That is, the superket after the map is found by multiplying the PTM by the superket before the map. This means if we can find the PTM for a given map (by finding the result of its application to the basis vectors) then we can find the result of the operation for *any* initial state.

The question is now how to transfer from density matrices and operators to superkets and superoperators. We use a decomposition method to *vectorise* the matrices.

That is, the superket is a *vector* of length $n^2$ which has elements

$$\ket{\rho}\rangle_i=\langle\braket{k|\rho}\rangle=\text{Tr }P_k\rho$$

Where $k\equiv P_k$. That is, decompose the matrix $\rho$ into the basis $P$, and then the superket represents these coefficients.

For example,

$$\ket{\rho}\rangle=\begin{pmatrix}0.5\\0\\0.2\\0.3\end{pmatrix}\implies \rho =0.5P_0+0P_1+0.2P_2+0.3P_3$$

### Key Properties

The PTM has a number of key properties which make it useful.

 - The elements satisfy $(R_\Lambda)_{ij}\in\mathbb{R}, (R_\Lambda)_{ij}\in[-1, 1]$
 - The PTM of some composite map is equal to the product of the individual PTMs, which makes simulating a circuit easy

Then, all maps $\Lambda$ must be CPTP (completely positive (positive probabilities) and trace-preserving (probabilities add to 1)). This puts two requirements on $R_\Lambda$:

$$\left(R_\Lambda\right)_{0j}=\delta_{0j}=(1,0,0,\dots,0)$$

The second requirement is that we require $\rho_\Lambda$ to be positive semi-definite, where

$$\rho_\Lambda=\frac{1}{d^2}\sum_{i,j=1}^{d^2}{(R_\Lambda)_{ij}P_j^T\otimes P_i}$$

The map may also be unital, which means that it maps the identity to the identity, or equivalently, it does not make a state more pure. This can be visualised as the requirement that the zero vector on the Bloch sphere (maximally mixed state) is not mapped to a nonzero vector (less mixed state). If the map is unital, we find that the first *column* of $R_\Lambda$ satisfies

$$R_{i0}=\delta_{i0}=\begin{pmatrix}1\\0\\0\\\vdots\\0\end{pmatrix}$$

## Errors

does error analysis require me to change from a PTM to a process matrix?

## Algorithm

The algorithm I will use is as follows

1. Set up a transmon with some defined parameters
2. Using pulse-level simulation, find the optimal pulse parameters to execute an X90 pulse. Initially start with $\ket{0}$, but once the pulse is near-optimal, simulate a random starting state
3. Calculate the PTM by running the pulse on all of the basis states
4. Verify the PTM by comparing the simulated and pulse-level result for some initial state
5. Analyse the PTM to correlate error sources and types?

Discuss this plan with Peter.

## Notes

The inner product of two vectors or matrices is defined as

$$\braket{A|B}\text{ or }\langle\braket{A|B}\rangle = \text{Tr } A^\dag B$$

And the expected value of any operator is

$$\braket{E}=\text{Tr }E\rho$$