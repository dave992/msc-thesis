import numpy as np

class PID():
    "PID Controller"
    
    def __init__(self, Kp, Ki = 0, Kd = 0, dt = 1/100):
        
        # Set input parameters
        self.dt = dt
        self.set_gains(float(Kp), float(Ki), float(Kd))

        # Set initial PID values
        self.P = np.array([0.0])
        self.I = np.array([0.0])
        self.D = np.array([0.0])
        self.last_error = np.array([0.0])
            
    def __call__(self, error):
        # Proportional Term
        self.P = self.Kp*error
        
        # Integral Term
        self.I = self._I(error)
        
        # Derivative Term
        self.D = self._D(error)
        self.last_error = error
        
        return self.P + self.I + self.D

    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        if Ki == 0.0:
            self._I = lambda error: 0.0
        else:
            self._I = lambda error: self.I + self.Ki*error*self.dt
            
        if Kd == 0.0:
            self._D = lambda error: 0.0
        else: 
            self._D = lambda error: -self.Kd*(error - self.last_error)/self.dt