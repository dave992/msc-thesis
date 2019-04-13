import math

import numpy as np
import scipy.signal as signal

from utils.pid import PID

# Standard signal components
cos_p20         = lambda dt: np.cos(1/10*np.pi*np.arange(0, 20, dt))
cos_p10         = lambda dt: np.cos(1/5.0*np.pi*np.arange(0, 10, dt))
cos_p05         = lambda dt: np.cos(1/2.5*np.pi*np.arange(0, 5, dt))
doublet2        = lambda dt: np.hstack([np.zeros(int(1/dt)), np.ones(int(2/dt)), -np.ones(int(4/dt)), np.ones(int(2/dt)), np.zeros(int(2/dt))])
doublet1        = lambda dt: np.hstack([np.zeros(int(0.5/dt)), np.ones(int(1/dt)), -np.ones(int(1/dt)), np.zeros(int(0.5/dt))])
doublet321_2    = lambda dt: np.hstack([np.zeros(int(1/dt)), np.ones(int(2/dt)), -np.ones(int(1.5/dt)), np.ones(int(0.5/dt)), -np.ones(int(0.5/dt)), np.zeros(int(1/dt))])
doublet321_3    = lambda dt: np.hstack([np.zeros(int(1/dt)), np.ones(int(3/dt)), -np.ones(int(2/dt)), np.ones(int(1/dt)), -np.ones(int(1/dt)), np.zeros(int(1/dt))])
doublet2_zeros  = lambda dt: np.zeros_like(doublet2(dt))

excitation      = lambda dt: np.vstack([-1.0*signal.sweep_poly(np.arange(0, 20, dt) , 0.075*np.poly1d([10], True)), 
                                        -2.5*np.sin(1/2.5*np.pi*np.arange(0, 20, dt)),
                                        0.*signal.chirp(np.arange(0, 20, dt), 0.25, 20, 1, phi=90)])


# Test maneuvers
omega_r1            = np.pi/60      # Rate 1 turn
omega_r2            = np.pi/30      # Rate 2 turn
omega_r3            = np.pi/15      # Rate 3 turn
zeros               = lambda T, dt: np.zeros(int(T/dt))     
ones                = lambda T, dt: np.ones(int(T/dt))
bank_angle          = lambda V, ROT: np.clip(np.arccos(1/np.sqrt((V**2/(9.81*(V/ROT)))**2 + 1)),-np.deg2rad(25),np.deg2rad(25))     # Calculate bank angle [rad] with desired rate of turn (ROT) [rad/s]

def climb_decent_profile(time_steps, dt, climb_rate=5, alt_0 = 0):

    # Construct altitude profile
    alt = [zeros(10,dt)]
    alt += [np.linspace(0.0, 45*climb_rate, int(45/dt))]
    alt += [45*climb_rate*ones(10,dt)]
    alt += [np.linspace(45*climb_rate, 90*climb_rate, int(45/dt))]
    alt += [90*climb_rate*ones(10,dt)]
    alt += [np.linspace(90*climb_rate, 45*climb_rate, int(45/dt))]
    alt += [45*climb_rate*ones(10,dt)]
    alt += [np.linspace(45*climb_rate, 0.0, int(45/dt))]
    alt += [zeros(10,dt)]
    alt = np.hstack(alt)
    alt = np.tile(alt, math.ceil(time_steps/alt.size))[:time_steps]

    # Construct bank angle profile
    phi = np.zeros_like(alt)
    
    return alt + alt_0, phi

def hold_profile(V, time_steps, dt, alt_0 = 0):

    # Construct bank angle profile
    phi = [zeros(20,dt)]
    phi += [ones(30,dt)*bank_angle(V, omega_r1)]
    phi += [ones(15,dt)*bank_angle(V, omega_r2)]
    phi += [zeros(20,dt)]
    phi = np.hstack(phi)
    phi = np.tile(phi, math.ceil(time_steps/phi.size))[:time_steps]

    # Construct altitude profile
    alt = np.zeros_like(phi)

    return alt + alt_0, phi

def spiral_profile(V, time_steps, dt, climb_rate = 5, alt_0 = 0):
    alt = [zeros(10, dt)]
    alt += [np.linspace(0.0, 120.0*climb_rate, int(120/dt))] 
    alt += [120.0*climb_rate*ones(10, dt)]
    alt += [np.linspace(120.0*climb_rate, 0.0, int(120/dt))]
    alt += [zeros(10, dt)]
    alt = np.hstack(alt) 
    alt = np.tile(alt, math.ceil(time_steps/alt.size))[:time_steps]

    # Construct bank angle profile
    phi = [zeros(10,dt)]
    phi += [ones(120,dt)*bank_angle(V, omega_r1)]
    phi += [zeros(10,dt), -ones(120,dt)*bank_angle(V, omega_r1)]
    phi += [zeros(10,dt)]
    phi = np.hstack(phi)
    phi = np.tile(phi, math.ceil(time_steps/phi.size))[:time_steps]

    return alt + alt_0, phi

def experiment_profile(dt, alt_0 = 0):
    # Altitude Profile
    alt_train   = [zeros(4, dt), np.arange(0.0, 20.0, 5*dt), np.arange(20.0, -20.0, -5*dt), np.arange(-20.0, 0.0, 5*dt), zeros(10, dt)]
    alt_expl    = [zeros(5, dt), np.arange(0.0, 750.0, 5*dt), 750*ones(30, dt), np.arange(750.0, 550.0, -5*dt), 550*ones(45, dt)]
    alt         = np.hstack(alt_train + alt_expl)

    # Bank Profile
    phi_train   = [np.arange(0.0, 20.0, 5*dt), 20*ones(2, dt), np.arange(20.0, -20.0, -5*dt), -20*ones(2, dt), np.arange(-20.0, 0.0, 5*dt), zeros(10, dt)]
    phi_expl    = [zeros(15, dt), np.arange(0.0, 25.0, 2.5*dt), 25*ones(30, dt), np.arange(25.0, 0.0, -2.5*dt), 
                    zeros(20, dt), np.arange(0.0, -25, -2.5*dt), -25*ones(30, dt), np.arange(-25.0, 0.0, 2.5*dt), 
                    zeros(20, dt), np.arange(0.0, -20, -2.5*dt), -20*ones(34, dt), np.arange(-20, 20, 2.5*dt), 20*ones(34, dt), np.arange(20,0,-2.5*dt), zeros(15, dt)]
    phi         = np.hstack(phi_train + phi_expl)

    return alt + alt_0, np.deg2rad(phi)

# Standard signals
doublet_signal         = lambda dt: np.hstack([doublet2(dt), doublet2_zeros(dt), -doublet2(dt), doublet2_zeros(dt)])   # ~ 17 sec
doublet_signal2        = lambda dt: np.hstack([doublet2_zeros(dt), doublet2(dt), doublet2_zeros(dt), -doublet2(dt)])

def overactuation_gradient_correction(gradients, actions, actions_clipped):
    is_overactuated = (actions - actions_clipped > 0.0) + (actions - actions_clipped < 0.0)
    gradients[:,:,is_overactuated.flatten()] = (-0.002*(actions - actions_clipped)[is_overactuated]/abs(actions - actions_clipped)[is_overactuated]).reshape(1,1,-1)
    return gradients

class LowPassFilter():
    " First order low pass filter. "
    default_time_constant = 0.1

    def __init__(self, dt, time_constant = None):
        " Initialize the low pass filter "
        self.time_constant = time_constant if time_constant is not None else self.default_time_constant
        self.alpha = dt/(dt + self.time_constant)
        self.reset()

    def __call__(self, x):
        " Redirect to filter() "
        return self.filter(x)

    def filter(self, signal):
        " Apply the filter to the signal. "
        y = np.zeros_like(signal)
        for i, x in np.ndenumerate(signal):
            y[i] = self.alpha*x + (1 - self.alpha)*self.y
            self.y = y[i].copy()
        return y

    def reset(self):
        self.y = np.zeros(1)

class AltitudeController():
    Kp_h                = 1/5*(np.pi/180)      # [rad/m]   1/10: 100 m > 10 deg; pi/180: deg > rad
    Ki_h                = 1/40*(np.pi/180)    # [rad/m]   Kp >> Ki, P-gain is chosen to be dominant, as steady state error is expected to be small. 
    Kp_theta            = 1.5                  # [s^-1]    0.5: 10 deg > 5 deg/s 
    max_altitude_error  = 20                   # [m]   

    def __init__(self, dt, Kp_h = None, Kp_theta = None, Ki_h = None):
        " Initialize the Altitude controller "
        if Kp_h is not None:
            self.Kp_h = Kp_h
        if Ki_h is not None:
            self.Ki_h = Ki_h
        if Kp_theta is not None:
            self.Kp_theta = Kp_theta

        self.altitude_pid   = PID(Kp=self.Kp_h, Ki=self.Ki_h, Kd=0.0, dt=dt)
        self.pitch_pid      = PID(Kp=self.Kp_theta, Ki=0.0, Kd=0.0, dt=dt)

    def cmd(self, altitude_ref, altitude, pitch):
        " Generate the pitch rate reference using the outer loop controllers "
        # Altitude loop (PI-control) > Pitch loop (P-control):
        pitch_ref   = self.altitude_pid(np.clip(altitude_ref - altitude, -self.max_altitude_error, self.max_altitude_error))
        q_ref       = self.pitch_pid(pitch_ref - pitch)
        return q_ref

    def set_gains(self, Kp_h = None, Ki_h = None, Kp_theta = None):
        " Set the gains of the PID controllers. "
        if Kp_h is not None:
            self.Kp_h = Kp_h
            self.altitude_pid.Kp = Kp_h
        if Ki_h is not None:
            self.Ki_h = Ki_h
            self.altitude_pid.Ki = Ki_h
        if Kp_theta is not None:
            self.Kp_theta = Kp_theta
            self.pitch_pid.Kp = Kp_theta
        
class RollAngleController():
    Kp_p                    = 0.75           # [s^-1]
    Ki_p                    = 0.05
    Kd_p                    = 0.001
    max_roll_angle_error    = np.pi/6       # 30 [deg]

    def __init__(self, dt, Kp_p = None, Ki_p = None, Kd_p = None):
        " Initialize the Roll Angle controller. "
        if Kp_p is None:
            Kp_p = self.Kp_p
        if Ki_p is None:
            Ki_p = self.Ki_p
        if Kd_p is None:
            Kd_p = self.Kd_p
        self.roll_pid = PID(Kp=Kp_p, Ki=0.0, Kd=0.0, dt=dt)

    def cmd(self, phi_ref, phi, error = None):
        " Generate the roll rate reference using the outer loop contoller. "
        if phi_ref is None and phi is None and error is not None:
            return self.roll_pid(np.clip(error, -self.max_roll_angle_error, self.max_roll_angle_error))
        elif phi_ref is not None and phi is not None and error is None:
            return self.roll_pid(np.clip(phi_ref - phi, -self.max_roll_angle_error, self.max_roll_angle_error))
        else:
            raise ValueError("Either use both the 'phi_ref' and 'phi' arguments, or only the 'error' argument.")

    def set_gains(self, Kp_p):
        " Set the gains of the PID controller. "
        self.Kp_p           = Kp_p
        self.roll_pid.Kp    = Kp_p
