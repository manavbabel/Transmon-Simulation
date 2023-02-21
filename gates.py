# set up the gate class

import numpy as np
from shapes import *


class MyGate:

    def __init__(self, axis, angle):

        # angle is given in degrees

        if axis not in ["X", "Y"]:
            raise ValueError("axis must be either 'X' or 'Y'")
        
        self.name = "R"+axis
        self.axis = axis
        self.angle = float(angle)

        self.phase = 0
        if self.axis == "Y":
            self.phase -= np.sign(self.angle) * np.pi/2
        elif self.name == "X" and self.angle == -90:
            self.phase = -np.pi

        self.optimal_parameters = None
        self.parameter_noise = None

    def H1_coeffs(self, t, args):

        try:
            offset=args["offset"]
        except KeyError:
            offset=0
        
        A,Γ,ω,A_DRAG = args["A"], args["Γ"], args["ω"], args["A_DRAG"]

        return sinusoid(t,1,ω,self.phase,0) * blackman(t-offset,A,Γ,A_DRAG)
    
    # the below is not used in anything yet    
    def get_pulse(self, dt, offset):

        t_actual = np.arange(start=offset, stop=offset+self.optimal_parameters["Γ"], step=dt)

        def H1_coeffs_partial(t, args):

            A,Γ,ω,A_DRAG = args["A"], args["Γ"], args["ω"], args["A_DRAG"]
            return sinusoid(t,1,ω,self.phase,0) * blackman(t-offset,A,Γ,A_DRAG)

        return t_actual, H1_coeffs_partial