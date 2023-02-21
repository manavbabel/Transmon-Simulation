import numpy as np
import sympy

# defines the letters used later as sympy symbols
x,A,ω,φ,y0,Γ,A_DRAG = sympy.symbols("x,A,ω,φ,y0,Γ,A_DRAG")

# window function for a Blackman pulse
# func is a sympy symbolic expression
def window(func):
    return sympy.Piecewise(
        (0, x>Γ),
        (0, x<0),
        (func, True)
    )

# shape functions

sinusoid_sym = A*sympy.cos(ω*x-φ)+y0

# CHECK THE SHIFTS FOR THE DRAG ADDITIONS
# for Gaussian it's pi/2, but for Blackman Γ/2 seems to work instead?
# blackman_sym = A*(1-(0.42 - 0.5 * sympy.cos(2*sympy.pi*(x-μ)/M) + 0.08 * sympy.cos(4*sympy.pi*(x-μ)/M)))
blackman_sym = A*(0.42 - 0.5 * sympy.cos(2*sympy.pi*(x-Γ)/Γ) + 0.08 * sympy.cos(4*sympy.pi*(x-Γ)/Γ))
blackman_sym = window(blackman_sym + A_DRAG * sympy.diff(blackman_sym, x).subs(x, x-Γ/2))

# converts the symbolic expressions to actual functions

sinusoid_func = sympy.lambdify([x,A,ω,φ,y0], sinusoid_sym, "numpy")
blackman_func = sympy.lambdify([x,A,Γ,A_DRAG], blackman_sym, "numpy")

# sinusoid
def sinusoid(x_val,A_val,ω_val,φ_val,y0_val):
    return sinusoid_func(x_val,A_val,ω_val,φ_val,y0_val)

# blackman
# as defined on the numpy page
# Γ_val is the FWHM, while M is the Blackman parameter
def blackman(x_val,A_val,Γ_val,A_DRAG_val): 
    return blackman_func(x_val,A_val,Γ_val,A_DRAG_val)