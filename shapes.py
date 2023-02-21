# shape functions

import sympy

# defines the letters used later as sympy symbols
x,A,ω,φ,Γ,A_DRAG = sympy.symbols("x,A,ω,φ,Γ,A_DRAG")

# window function for a Blackman pulse
# func is a sympy symbolic expression
# sets the function to zero except in the range of inputs between 0 and the variable Γ
def window(func):
    # should the other two be 0 or very very small?
    return sympy.Piecewise(
        (1e-10, x>Γ),
        (1e-10, x<0),
        (func, True)
    )

# shape functions
# should this be a +φ or a -φ? i think it was originally +φ
sinusoid_sym = sympy.cos(ω*x+φ)

# CHECK THE SHIFTS FOR THE DRAG ADDITIONS
# for Gaussian it's pi/2, but for Blackman Γ/2 seems to work instead?
blackman_sym = A*(0.42 - 0.5 * sympy.cos(2*sympy.pi*(x-Γ)/Γ) + 0.08 * sympy.cos(4*sympy.pi*(x-Γ)/Γ))
blackman_sym = window(blackman_sym + A_DRAG * sympy.diff(blackman_sym, x).subs(x, x-Γ/2))

# converts the symbolic expressions to actual functions
sinusoid = sympy.lambdify([x,ω,φ], sinusoid_sym, "numpy")
blackman = sympy.lambdify([x,A,Γ,A_DRAG], blackman_sym, "numpy")

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