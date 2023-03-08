# shape functions

# the upper one is the updated one
# which uses
# https://arxiv.org/pdf/1011.1949.pdf
# the lower, commented one, is the original

# NEWER H1_COEFFS
import sympy as sym
import numpy as np

A,x,τ,Γ = sym.symbols("A,x,τ,Γ")

Γ = τ/4

Ωg = A * (sym.pi/2) * (sym.exp((-1*(x-τ/2)**2)/(2*Γ**2))-sym.exp((-1*τ**2)/(8*Γ**2))) / (sym.sqrt(2*sym.pi*Γ**2)*sym.erf(τ/(sym.sqrt(8)*Γ))-τ*sym.exp((-1*τ**2)/(8*Γ**2)))

Ωgprime = sym.diff(Ωg,x)

padding = 0.002
Γ = τ-2*padding

Ωb = A * (50*sym.pi/(42*Γ)) * (0.42 - 0.5*sym.cos(2*sym.pi*(x-padding)/Γ) + 0.08*sym.cos(4*sym.pi*(x-padding)/Γ))

Ωb = sym.Piecewise(
    (0,x<padding),
    (0,x>τ-padding),
    (Ωb, True))

Ωbprime = sym.diff(Ωb,x)

Ω = sym.lambdify([x,A,τ], Ωb, modules=['numpy', 'math'])
Ωprime = sym.lambdify([x,A,τ], Ωbprime, modules=['numpy', 'math'])

def H1_coeffs(t,args):

    A,τ,λ,α,ω = args["A"], args["τ"], args["λ"], args["α"], args["ω"]

    offset = args.get("offset", 0)
    φ = args.get("φ", 0)

    Ωx = Ω(t-offset,A,τ)
    Ωy = -λ*Ωprime(t-offset,A,τ)/α
    
    return (Ωx*np.cos(ω*t+φ)) + (Ωy*np.sin(ω*t+φ))

"""
# OLDEST H1_COEFFS
import sym

# defines the letters used later as sym symbols
x,A,ω,φ,Γ,A_DRAG = sym.symbols("x,A,ω,φ,Γ,A_DRAG")

# window function for a Blackman pulse
# func is a sym symbolic expression
# sets the function to zero except in the range of inputs between 0 and the variable Γ
def window(func):
    # should the other two be 0 or very very small?
    return sym.Piecewise(
        (1e-10, x>Γ),
        (1e-10, x<0),
        (func, True)
    )

# shape functions
# ω is in rads per second
sinusoid_sym = sym.cos(ω*x+φ)

blackman_sym = A*(0.42 - 0.5 * sym.cos(2*sym.pi*(x-Γ)/Γ) + 0.08 * sym.cos(4*sym.pi*(x-Γ)/Γ))
blackman_sym = window(blackman_sym + A_DRAG * sym.diff(blackman_sym, x).subs(x, x-Γ/2))

# converts the symbolic expressions to actual functions
sinusoid = sym.lambdify([x,ω,φ], sinusoid_sym, "numpy")
blackman = sym.lambdify([x,A,Γ,A_DRAG], blackman_sym, "numpy")

def H1_coeffs(t, args):
    
    if "φ" not in args.keys():
        φ = 0
    else:
        φ = args["φ"]
    if "offset" not in args.keys():
        offset = 0
    else:
        offset = args["offset"]

    A,Γ,ω,A_DRAG = args["A"], args["Γ"], args["ω"], args["A_DRAG"]
    return sinusoid(t,ω,φ) * blackman(t-offset,A,Γ,A_DRAG)
"""