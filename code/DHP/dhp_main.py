import datetime
import importlib
import math
import os
import pathlib
import sys

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

# Custom classes
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import utils
from utils import phlab, LowPassFilter, AltitudeController, RollAngleController
from utils.phlab import loc
from agents import dhp as DHP
from agents import model 

print('Imports done')

##### Flags #####
LOG                 = False
SAVE                = False
REPEAT_UPDATE       = True
PLOT                = True
FLIGHT_CONDITION    = 3


##### Locations #####
TENSORBOARD_DIR = './logs/tensorboard/DHP/'
CHECKPOINT_DIR = './logs/checkpoint/DHP/'
FRAMEWORK_NAME = 'dhp_citation_presentation_fc' + str(FLIGHT_CONDITION)

##### Tracking & Mode Settings #####
MODE                    = phlab.LATLON
TRACK                   = ['p', 'q', 'beta']
EXCLUDE                 = []                                            # Exclude: 'V' Airspeed, 'h' Altitude, or None if none is excluded
ID                      = phlab.get_tracking_id(TRACK, MODE, EXCLUDE)   # Get unique ID for mode, tracking and excluded states
ID_LATLON               = phlab.ID_LATLON
STATE_ERROR_WEIGHTS     = phlab.get_state_error_weights(ID)             # Cost function error weights 
TRACKED, phlab.command  = phlab.get_tracked_states(ID)                  # Tracked state list (Boolean)


##### Framework and Simulation set-up #####
# Simulation parameters
dt  = 0.02                                                              # [s] - Time step
T   = 300.                                                              # [s] - Episode Length
t   = np.arange(0, T, dt)                                               # [s] - Time vector
time_steps      = len(t)                                                # [-] - Number of timesteps
state_size      = phlab.get_citation_state_size(MODE)                   # [-] - Number of environment states
ac_state_size   = len(TRACKED)                                          # [-] - Number of actor critic environment states
action_size     = phlab.get_action_size(ID)                             # [-] - Number of environment actions
update_cycles   = 2 if REPEAT_UPDATE else 1                             # [-] - Number of internal update cycles

# Info messages
phlab.display_tracking_info(ID)
print('  states: ' + str(state_size) + '\n  ac_states: ' + str(ac_state_size) + '\n  actions: ' + str(action_size) + '\n')

# Initialize Citation and Maneuvers
citation                            = importlib.import_module('envs.phlab.fc' + str(FLIGHT_CONDITION) + '.citation_act')
citation.initialize()
x0                                  = phlab.get_initial_state(phlab.ID_LATLON, FLIGHT_CONDITION)                        # Initial state
cmd                                 = phlab.get_initial_trim(FLIGHT_CONDITION)                                          # Actuator positions for initial state
env                                 = np.zeros(9)                                                                       # Turbulence parameters (zeros is no turbulence)
failure                             = np.zeros(1)                                                                       # np.ones(1) for a stuck (at a fixed predeterminded position) aileron failure
alt_signal, phi_signal              = utils.experiment_profile(dt, alt_0= x0[:,phlab.state2loc(ID)['h'],:].squeeze())   # Reference altitude and bank angle to fly
excitation                          = np.deg2rad(utils.excitation(dt))                                                  # Excitation signal (for elevator and aileron)

# Create Agent
kwargs = {
    'input_size': [ac_state_size, np.sum(TRACKED)],                     # [Aircraft state size, Number of tracked states]
    'output_size': action_size,                                         # Actor output size (Critic output is dependend only on aircraft state size)
    'hidden_layer_size': [50,50,50],                                    # List with number of nodes per layer, number of layers is variable
    'kernel_stddev': 0.1,                                               # Standard deviation used in the truncated normal distribution to initialize all parameters
    'lr_critic': 0.1,                                                   # Learn rate Critic
    'lr_actor': 0.05,                                                   # Learn rate Actor
    'gamma': 0.4,                                                       # Discount factor
    'use_bias': False,                                                  # Use bias terms (only if you train using batched data)
    'split': True,                                                      # Split architechture of the actor, if False, a single fully connected layer is used.
    'target_network': True,                                             # Use target networks 
    'tau': 0.001,                                                       # Target network time constant 
    'activation': tf.nn.relu,                                           # Activation function of the hidden layers
    'log_dir': TENSORBOARD_DIR,                                         # Where to save checkpoints
    'use_delta': (False, TRACKED)                                       # (True, TRACKED) used 's = [x, (x - x_ref)]' || (False, None) uses 's = [x, x_ref]' 
}
agent               = DHP.Agent(**kwargs)                               
agent.trim          = cmd[0:action_size].copy()                         # Set inital trim as ofset to the actor (optional)
lr_critic           = kwargs['lr_critic']                               # Extract default learn rate, as the update can accept a custom learn rate to implement learn rate scheduling 
lr_actor            = kwargs['lr_actor']

# Create aircraft model
ac_kwargs = {
        # Arguments for all model types
        'state_size': ac_state_size,
        'action_size': action_size,
        'predict_delta': False,
        # Neural Network specific args:
        'hidden_layer_size': [100, 100, 100],
        'activation': tf.nn.relu,
        # RLS specific args:
        'gamma': 0.9995,
        'covariance': 100,
        'constant': True,
        # LS specific args:
        'buffer_length': 10
}
# ac_model = model.NeuralNetwork('./logs/models/nn/citation_normalized', **ac_kwargs)
ac_model = model.RecursiveLeastSquares(**ac_kwargs)

# Create altitude and roll angle (outer loop) controllers
h_controller    = AltitudeController(dt = dt)
p_controller    = RollAngleController(dt = dt)

# Create DataFrame & Storage lists
data = pd.DataFrame({
    'Time': t,
    'State': [np.zeros(state_size)]*time_steps, 
    'Action': [0.]*time_steps, 
    'Cost': [0.]*time_steps, 
    'Next State':[np.zeros(state_size)]*time_steps,
    'Reference State': [np.zeros(agent.input_size)]*time_steps})
list_X, list_A, list_C, list_X_ref      = ([] for _ in range(4))
list_X_predict, list_C_trained          = ([] for _ in range(2))
list_critic_grad,  list_action_grad     = ([] for _ in range(2))
list_F, list_G, list_RLSCov, e_model    = ([] for _ in range(4))
list_T                                  = []
list_vars_actor                         = []

##### Run training #####
with tqdm(range(time_steps)) as tqdm_it:
    
    # Initialize log features
    run_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Initialize parameters
    X       = phlab.normalize(phlab.get_initial_state(ID, FLIGHT_CONDITION), ID)
    X_full  = phlab.get_initial_state(phlab.ID_LATLON, FLIGHT_CONDITION)
    X_ref   = np.zeros_like(X)
    X_fref  = np.zeros_like(X_full)
    P       = np.diag(TRACKED).astype(float)
    Q       = np.diag(STATE_ERROR_WEIGHTS)
    list_X.append(X_full)

    # Initialize variables for efficiency
    loc_ref2fref = [phlab.state2loc(ID_LATLON)[x] for x in phlab.states[ID]]
    
    # Start simulation and training loop
    for i in tqdm_it:
        # Sample reference signal
        phi_ref             = phi_signal[i]
        alt_ref             = alt_signal[i]
        p_ref               = np.clip(p_controller.cmd(phi_ref, X_full[:,loc('phi', ID_LATLON)]), np.deg2rad(-10), np.deg2rad(10))
        q_ref               = np.clip(h_controller.cmd(alt_ref, X_full[:,loc('h', ID_LATLON)], X_full[:,loc('theta', ID_LATLON)] - X_full[:,loc('alpha', ID_LATLON)]), np.deg2rad(-10), np.deg2rad(10))
        beta_ref            = 0.0
        R_sig               = np.array([p_ref, q_ref, beta_ref])[0:action_size].reshape([1,-1,1])
        X_ref[:,TRACKED,:]  = R_sig

        # Store full size reference
        X_fref[:, loc_ref2fref,:]           = X_ref 
        X_fref[:, loc('phi', ID_LATLON),:]  = phi_ref
        X_fref[:, loc('h', ID_LATLON),:]    = alt_ref
        list_X_ref.append(X_fref.copy())
        
        ### Update Agent ###
        j = 0
        while j < update_cycles:

            # Next state prediction
            action          = agent.action(X, reference = R_sig).reshape([1,-1,1])
            action_clipped  = np.clip(action, np.array([[[-0.35], [-0.65], [-0.38]]]), np.array([[[0.26], [0.65], [0.38]]]))
            X_next_pred     = ac_model.predict(X, action_clipped).reshape([1,-1,1])
            
            # Cost prediction
            e               = np.matmul(P, X_next_pred - X_ref) 
            cost            = np.matmul(np.matmul(e.transpose(0,2,1), Q), e)
            dcostdx         = np.matmul(2*np.matmul(e.transpose(0,2,1), Q), P)
            
            # Critic
            dactiondx       = agent.gradient_actor(X, reference = R_sig)
            lmbda           = agent.value_derivative(X, reference = R_sig)
            target_lmbda    = agent.target_value_derivative(X_next_pred, reference = R_sig)
            A               = ac_model.gradient_state(X, action)
            B               = ac_model.gradient_action(X, action)
            grad_critic     = lmbda - np.matmul(dcostdx + agent.gamma*target_lmbda, A + np.matmul(B, dactiondx))            
            grad_critic     = np.clip(grad_critic, -0.2, 0.2)
            agent.update_critic(X, reference = R_sig, gradient = grad_critic, learn_rate=lr_critic)

            # Actor
            lmbda       = agent.value_derivative(X_next_pred, reference = R_sig)
            grad_actor  = np.matmul(dcostdx + agent.gamma*lmbda, B)
            # grad_actor  = np.clip(grad_actor, -0.1, 0.1)
            # grad_actor  = utils.overactuation_gradient_correction(gradients=grad_actor, actions=action, actions_clipped=action_clipped)
            agent.update_actor(X, reference = R_sig, gradient = grad_actor, learn_rate=lr_actor)
            
            # Loop management (for possible dynamic loop sizes)
            j += 1

        # Store intermediate results
        list_X_predict.append(X_next_pred)
        list_C_trained.append(cost.flatten())
        list_action_grad.append(grad_actor)
        list_critic_grad.append(grad_critic)
        list_F.append(A.flatten().copy())
        list_G.append(B.flatten().copy())
        list_RLSCov.append(ac_model.cov.copy())

        ### Run environment ###
        action      = agent.action(X, reference = R_sig)
        if i < 1000:
            action += excitation[:,i].reshape(action.shape)
        action      = np.clip(action, np.array([[-0.35, -0.65, -0.38]]), np.array([[0.26, 0.65, 0.38]]))
        cmd         = phlab.command(cmd, action)
        X_cit, _    = citation.step(cmd, env, failure)
        X_full      = phlab.cit2latlon(X_cit)
        X_next      = phlab.full2mode(X_cit, ID).reshape([1,-1,1])
        model_error = ((X_next_pred - X_next)**2).mean()
        list_X.append(X_full)
        list_A.append(cmd[:3].copy().reshape([1,-1,1]))
        e_model.append(model_error)
        list_T      += [X_cit[29]]

        ### Real Cost ###
        e           = np.matmul(P, (X_next - X_ref)) 
        cost        = np.matmul(np.matmul(e.transpose(0,2,1), Q), e)
        list_C.append(cost)

        ### Update Model ###
        ac_model.update(X, action, X_next)

        ### Bookkeeping ###
        X = X_next.copy()

        ### Log & Save ###
        P_actor = agent.session.run(agent.vars_actor)
        list_vars_actor.append(P_actor)
        if SAVE and i % int(.05*time_steps) == 0: # Note: Move this to before the update if the initialized network needs to be saved, as here the network has already been updated! 
            agent.save(file_path = CHECKPOINT_DIR + FRAMEWORK_NAME +'_' + run_time + '/ckpt', global_step = i)

        ### NaN detection ###                   Failure criteria as a result of exploding parameters/outputs
        if np.isnan(action).any():
            print('\n\nAborting training: action = NaN\nTimestep: {} (= {} s)'.format(i, i*dt))

            # Fill all lists to ensure plots and saves still work 
            time_steps_left = time_steps - (i + 1)          
            list_X          += [np.nan*np.ones_like(X_full)]*time_steps_left
            list_A          += [np.nan*np.ones([1,3,1])]*time_steps_left
            list_C          += [np.nan*np.ones_like(cost)]*time_steps_left
            list_C_trained  += [np.nan*np.ones_like(cost.flatten())]*time_steps_left
            list_X_ref      += [np.nan*np.ones_like(X_full)]*time_steps_left
            e_model         += [np.nan*np.ones_like(((X_next_pred - X_next)**2).mean())]*time_steps_left
            list_action_grad+= [np.nan*np.ones_like(list_action_grad[0])]*time_steps_left
            list_critic_grad+= [np.nan*np.ones_like(list_critic_grad[0])]*time_steps_left
            list_vars_actor += [np.nan*np.ones_like(list_vars_actor[0])]*time_steps_left
            list_F          += [np.nan*np.ones_like(list_F[0])]*time_steps_left
            list_G          += [np.nan*np.ones_like(list_G[0])]*time_steps_left
            list_RLSCov     += [np.nan*np.ones_like(list_RLSCov[0])]*time_steps_left
            
            tqdm_it.close()
            break

# Store Data
data['State'] = list_X[:-1]
data['Action'] = list_A
data['Cost'] = list_C
data['Next State'] = list_X[1:]
data['Reference State'] = list_X_ref

# Extra dataframe to store gradients (from the last update cycle)
df = pd.DataFrame({'actor_grad': list_action_grad, 'critic_grad': list_critic_grad, 'actor_vars': list_vars_actor})
df.to_pickle('./logs/grads_and_vars_fc3.pkl')

### Run data ####
if SAVE:
    path = './logs/data/DF_' + FRAMEWORK_NAME + '_' + run_time + '.pkl'
    print('Saved data to:', path)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True) # pylint: disable=E1101
    data.to_pickle(path)

### Plot the episode ###
if PLOT: 
    X       = np.vstack(data['State'].values)
    X_pred  = np.vstack(list_X_predict)
    X_ref   = np.vstack(data['Reference State'].values)
    A       = np.vstack(data['Action'].values)
    A       = np.clip(A, np.deg2rad(-40), np.deg2rad(40))
    F       = np.vstack(list_F)
    G       = np.vstack(list_G)
    RLSvar  = np.vstack([np.diagonal(x) for x in list_RLSCov])
    RLSCov  = np.vstack([x.flatten() for x in list_RLSCov])
    e_model = np.vstack(e_model)
    c_train = np.vstack(list_C_trained)

    # Extract state values for readability
    p       = X[:,phlab.loc('p', ID_LATLON)]
    q       = X[:,phlab.loc('q', ID_LATLON)]
    r       = X[:,phlab.loc('r', ID_LATLON)]
    V       = X[:,phlab.loc('V', ID_LATLON)]
    alpha   = X[:,phlab.loc('alpha', ID_LATLON)]
    beta    = X[:,phlab.loc('beta', ID_LATLON)]
    phi     = X[:,phlab.loc('phi', ID_LATLON)]
    theta   = X[:,phlab.loc('theta', ID_LATLON)]
    h       = X[:,phlab.loc('h', ID_LATLON)]

    # Extract references and predictions for readability
    p_ref   = X_ref[:,phlab.loc('p', ID_LATLON)]
    q_ref   = X_ref[:,phlab.loc('q', ID_LATLON)]
    phi_ref = X_ref[:,phlab.loc('phi', ID_LATLON)]
    alt_ref = X_ref[:,phlab.loc('h', ID_LATLON)]
    beta_ref= X_ref[:,phlab.loc('beta', ID_LATLON)]
    p_pred  = X_pred[:,phlab.loc('p', ID)]
    q_pred  = X_pred[:,phlab.loc('q', ID)]

    # Set seaborn style and figure parameters
    sns.set()
    sns.set(font_scale=1.25)
    plt.rcParams['figure.figsize'] = 25/2.54, 50/2.54

    ### Figure 1 ###
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(4,1,1)
    ax1.plot(t, q/np.pi*180., 'b-', label=r'$q$')  
    ax1.plot(t, q_ref/np.pi*180., 'r--', label=r'$q_{ref}$')  
    ax1.set_ylabel(r'Pitch Rate $[deg/s]$')
    ax1.legend(loc = 'upper right')

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, A[:,0]/np.pi*180., 'b-', label=r'$\delta_e$')  
    ax2.set_ylabel(r'Elevator $[deg]$')
    ax2.legend(loc = 'upper right')

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, data['Cost'], 'b-', label=r'$c_{actual}$')  
    ax3.plot(t, c_train, 'r--', label=r'$c_{train}$')  
    ax3.set_xlabel(r'$t [s]$')
    ax3.set_ylabel(r'Cost [-]')
    ax3.set_yscale('log')
    ax3.legend(loc = 'upper right')

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, e_model/np.pi*180., 'b-', label=r'$e_{model}$') 
    ax4.set_xlabel(r'$t [s]$')
    ax4.set_ylabel(r'Model Error $[-]$')
    ax4.set_yscale('log')
    ax4.legend(loc = 'upper right')

    ### Figure 2 ###
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(4,1,1)
    ax1.plot(t, alpha/np.pi*180., 'b-', label=r'$\alpha$')  
    ax1.set_ylabel(r'Angle of Attack $[deg]$')
    ax1.legend(loc = 'upper right')

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, theta/np.pi*180., 'b-', label=r'$\theta$')  
    ax2.set_ylabel(r'Pitch Angle $[deg]$')
    ax2.legend(loc = 'upper right')

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, V, 'b-', label=r'$V_{TAS}$')  
    ax3.plot(t, phlab.CITATION_V_MIN*np.ones_like(t), 'r--', label=r'$V_{min}$') 
    ax3.plot(t, phlab.CITATION_V_MAX*np.ones_like(t), 'r--', label=r'$V_{max}$') 
    ax3.set_ylabel(r'Airspeed [m/s]')
    ax3.legend(loc = 'upper right')

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, h, 'b-', label=r'$h_e$') 
    ax4.plot(t, alt_ref, 'r--', label=r'$h_{e_{ref}}$') 
    ax4.set_xlabel(r'$t [s]$')
    ax4.set_ylabel(r'Height [m]')
    ax4.legend(loc = 'upper right')

    ### Figure 3 ###
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(4,1,1)
    ax1.plot(t, p/np.pi*180., 'b-', label=r'$p$')  
    ax1.plot(t, p_ref/np.pi*180., 'r--', label=r'$p_{ref}$')  
    ax1.set_ylabel(r'Roll Rate $[deg/s]$')
    ax1.legend(loc = 'upper right')

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, A[:,1]/np.pi*180., 'b-', label=r'$\delta_a$')  
    ax2.set_ylabel(r'Aileron $[deg]$')
    ax2.legend(loc = 'upper right')

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, A[:,2]/np.pi*180., 'b-', label=r'$\delta_r$')  
    ax3.set_ylabel(r'Rudder $[deg]$')
    ax3.legend(loc = 'upper right')

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, phi/np.pi*180., 'b-', label=r'$\phi$')  
    ax4.plot(t, phi_ref/np.pi*180., 'r--', label=r'$\phi_{ref}$')  
    ax4.set_ylabel(r'Roll Angle $[deg]$')
    ax4.legend(loc = 'upper right')

    ### Figure 4 ###
    fig4 = plt.figure()
    ax1 = fig4.add_subplot(4,1,1)
    ax1.plot(t, beta/np.pi*180., 'b-', label=r'$\beta$')  
    ax1.plot(t, beta_ref/np.pi*180., 'r--', label=r'$\beta_{ref}$')  
    ax1.set_ylabel(r'Sideslip $[deg]$')
    ax1.legend(loc = 'upper right')

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, A[:,2]/np.pi*180., 'b-', label=r'$\delta_r$')  
    ax2.set_ylabel(r'Rudder $[deg]$')
    ax2.legend(loc = 'upper right')

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, np.vstack(list_action_grad).squeeze())  
    # ax3.plot(t, list_var_actor, 'r-', label=r'$actor_{var}$')  
    ax3.set_ylabel(r'Actor Network')

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, np.vstack(list_critic_grad).squeeze())
    # ax4.plot(t, list_var_critic, 'r-', label=r'$critic_{var}$')
    ax4.set_ylabel(r'Critic Network')

    ### Figure 5 ###
    fig5 = plt.figure()
    ax1 = fig5.add_subplot(4,1,1)
    ax1.plot(t, F)
    ax1.set_ylabel(r'$\frac{\partial x_{t+1}}{\partial x_t}$')

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, G)
    ax2.set_ylabel(r'$\frac{\partial x_{t+1}}{\partial u_t}$')

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, RLSvar)
    ax3.set_ylabel(r'Variance')
    ax3.set_yscale('log')

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, RLSCov)
    ax4.set_ylabel(r'Covariance')
    ax4.set_yscale('log')

    # fig6 = plt.figure()
    # ax6 = fig6.add_subplot(1,1,1)
    # ax6.plot(t, list_T)
    ### Plot figures ###
    fig1.align_labels()
    fig2.align_labels()
    fig3.align_labels()
    fig4.align_labels()
    # plt.tight_layout()
    plt.show()