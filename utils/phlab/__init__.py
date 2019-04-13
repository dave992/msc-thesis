import numpy as np

##### Modes #####
LON     = 'lon'
LAT     = 'lat'
LATLON  = 'latlon'

#### Unique tracking mode identifiers ####
ID_LON                      = 100
ID_Q_LON                    = 102
ID_Q_LON_EX_V               = 118
ID_Q_LON_EX_H               = 134
ID_Q_LON_EX_VH              = 150
ID_LAT                      = 200
ID_LATLON                   = 300
ID_P_LATLON                 = 301
ID_Q_LATLON                 = 302
ID_PQ_LATLON                = 303
ID_PQ_LATLON_EX_V           = 319
ID_PQ_LATLON_EX_H           = 335
ID_PQ_LATLON_EX_VH          = 351
ID_PQ_BETA_LATLON           = 311
ID_PQ_BETA_LATLON_EX_V      = 327
ID_PQ_BETA_LATLON_EX_H      = 343
ID_PQ_BETA_LATLON_EX_VH     = 359

#### Dictionaries ####
track_states = {ID_LON                  : [False]*5,
                ID_Q_LON                : [True] + [False]*4,
                ID_Q_LON_EX_V           : [True] + [False]*3,
                ID_Q_LON_EX_H           : [True] + [False]*3,
                ID_Q_LON_EX_VH          : [True] + [False]*2,
                ID_LAT                  : [False]*6,
                ID_LATLON               : [False]*9,
                ID_P_LATLON             : [True] + [False]*8,
                ID_Q_LATLON             : [False, True] + [False]*7,
                ID_PQ_LATLON            : [True]*2 + [False]*7,
                ID_PQ_LATLON_EX_V       : [True]*2 + [False]*6,
                ID_PQ_LATLON_EX_H       : [True]*2 + [False]*6,
                ID_PQ_LATLON_EX_VH      : [True]*2 + [False]*5,
                ID_PQ_BETA_LATLON       : [True]*2 + [False]*3 + [True] + [False]*3,
                ID_PQ_BETA_LATLON_EX_V  : [True]*2 + [False]*2 + [True] + [False]*3,
                ID_PQ_BETA_LATLON_EX_H  : [True]*2 + [False]*3 + [True] + [False]*2,
                ID_PQ_BETA_LATLON_EX_VH : [True]*2 + [False]*2 + [True] + [False]*2}

states       = {ID_LON                  : ['q', 'V', 'alpha', 'theta', 'h'],
                ID_Q_LON                : ['q', 'V', 'alpha', 'theta', 'h'],
                ID_Q_LON_EX_V           : ['q', 'alpha', 'theta', 'h'],
                ID_Q_LON_EX_H           : ['q', 'V', 'alpha', 'theta'],
                ID_Q_LON_EX_VH          : ['q', 'alpha', 'theta'],
                ID_LAT                  : ['p', 'r', 'V', 'beta', 'phi', 'h'],
                ID_LATLON               : ['p', 'q', 'r', 'V', 'alpha', 'beta', 'phi', 'theta', 'h'],
                ID_P_LATLON             : ['p', 'q', 'r', 'V', 'alpha', 'beta', 'phi', 'theta', 'h'],
                ID_Q_LATLON             : ['p', 'q', 'r', 'V', 'alpha', 'beta', 'phi', 'theta', 'h'],
                ID_PQ_LATLON            : ['p', 'q', 'r', 'V', 'alpha', 'beta', 'phi', 'theta', 'h'],
                ID_PQ_LATLON_EX_V       : ['p', 'q', 'r', 'alpha', 'beta', 'phi', 'theta', 'h'],
                ID_PQ_LATLON_EX_H       : ['p', 'q', 'r', 'V', 'alpha', 'beta', 'phi', 'theta'],
                ID_PQ_LATLON_EX_VH      : ['p', 'q', 'r', 'alpha', 'beta', 'phi', 'theta'],
                ID_PQ_BETA_LATLON       : ['p', 'q', 'r', 'V', 'alpha', 'beta', 'phi', 'theta', 'h'],
                ID_PQ_BETA_LATLON_EX_V  : ['p', 'q', 'r', 'alpha', 'beta', 'phi', 'theta', 'h'],
                ID_PQ_BETA_LATLON_EX_H  : ['p', 'q', 'r', 'V', 'alpha', 'beta', 'phi', 'theta'],
                ID_PQ_BETA_LATLON_EX_VH : ['p', 'q', 'r', 'alpha', 'beta', 'phi', 'theta']}     

idx          = {ID_LON                  : [1, 3, 4, 7, 9],
                ID_Q_LON                : [1, 3, 4, 7, 9],
                ID_Q_LON_EX_V           : [1, 4, 7, 9],
                ID_Q_LON_EX_H           : [1, 3, 4, 7],
                ID_Q_LON_EX_VH          : [1, 3, 7],
                ID_LAT                  : [0, 2, 3, 5, 6, 9],
                ID_LATLON               : [0, 1, 2, 3, 4, 5, 6, 7, 9],
                ID_P_LATLON             : [0, 1, 2, 3, 4, 5, 6, 7, 9],
                ID_Q_LATLON             : [0, 1, 2, 3, 4, 5, 6, 7, 9],
                ID_PQ_LATLON            : [0, 1, 2, 3, 4, 5, 6, 7, 9],
                ID_PQ_LATLON_EX_V       : [0, 1, 2, 4, 5, 6, 7, 9],
                ID_PQ_LATLON_EX_H       : [0, 1, 2, 3, 4, 5, 6, 7],
                ID_PQ_LATLON_EX_VH      : [0, 1, 2, 4, 5, 6, 7],
                ID_PQ_BETA_LATLON       : [0, 1, 2, 3, 4, 5, 6, 7, 9],
                ID_PQ_BETA_LATLON_EX_V  : [0, 1, 2, 4, 5, 6, 7, 9],
                ID_PQ_BETA_LATLON_EX_H  : [0, 1, 2, 3, 4, 5, 6, 7],
                ID_PQ_BETA_LATLON_EX_VH : [0, 1, 2, 4, 5, 6, 7]}

state2loc       = lambda id: dict(zip(states[id], range(len(states[id]))))
state2citloc    = lambda id: dict(zip(states[id], idx[id]))

##### Constants #####
CITATION_V_MIN = 40
CITATION_V_MAX = 200 
CITATION_H_MIN = 0
CITATION_H_MAX = 4000

#### Trim values ####
#   Speed   : 90    m/s     
#   Altitude: 2000  m 
#   Gamma   : -3    deg
CITATION_TRIM_CMD                   = np.array([-0.024761262011031245, 1.3745996716698875e-14, -7.371050575286063e-14, 0, 0, 0, 0, 0, 0.38576210972746433, 0.38576210972746433])
CITATION_INITIAL_STATE              = np.array([0, 0, 0, 90, 0.05597573599847709, -4.858931351206341e-14, 0, 0.0036757359984770895, 0, 2000.])

#   Speed   : 140   m/s     
#   Altitude: 5000  m 
#   Gamma   : 0     deg
CITATION_TRIM_CDM_FC0               = np.array([-0.00240955080071394, -6.11714329886771e-15, 3.43481275701017e-14, 0, 0, 0, 0, 0, 0.795917671140022, 0.795917671140022])
CITATION_INITIAL_STATE_FC0          = np.array([0, 0, 0, 140, 0.0183270437823565, 2.27893870929102e-14, 0, 0.0183270437823565, 0, 5000, 0, 0])

#   Speed   : 90    m/s     
#   Altitude: 5000  m 
#   Gamma   : 0     deg
CITATION_TRIM_CDM_FC1               = np.array([-0.0484190827034197, 8.11078054777134e-16, -5.38618504871800e-15, 0, 0, 0, 0, 0, 0.569141981013629, 0.569141981013629])
CITATION_INITIAL_STATE_FC1          = np.array([0, 0, 0, 90, 0.0862999595131081, -3.80012014686269e-15, 0, 0.0862999595131081, 0, 5000, 0, 0])

#   Speed   : 140   m/s     
#   Altitude: 2000  m 
#   Gamma   : 0     deg
CITATION_TRIM_CDM_FC2               = np.array([0.00670896101742224, 7.09979497727021e-13, -3.81667948902778e-12, 0, 0, 0, 0, 0, 0.817842026848348, 0.817842026848348])
CITATION_INITIAL_STATE_FC2          = np.array([0, 0, 0, 140, 0.00628792484724130, -2.57977378078467e-12, 0, 0.00628792484724130, 0, 2000, 0, 0])

#   Speed   : 90    m/s     
#   Altitude: 2000  m 
#   Gamma   : 0     deg
CITATION_TRIM_CDM_FC3               = np.array([-0.0284819968588408, -1.22765833074699e-12, 7.07281498315080e-12, 0, 0, 0, 0, 0, 0.575698826704105, 0.575698826704105])
CITATION_INITIAL_STATE_FC3          = np.array([0, 0, 0, 90, 0.0562379523838900, 4.55901311597514e-12, 0, 0.0562379523838900, 0, 2000, 0, 0])

def get_initial_trim(flight_condition = 3):
    if flight_condition == 0:
        trim_cmd        = CITATION_TRIM_CDM_FC0
    if flight_condition == 1:
        trim_cmd        = CITATION_TRIM_CDM_FC1
    if flight_condition == 2:
        trim_cmd        = CITATION_TRIM_CDM_FC2
    if flight_condition == 3:
        trim_cmd        = CITATION_TRIM_CDM_FC3
    return trim_cmd

##### Functions #####
def cit2latlon(state):
    return state[idx[ID_LATLON]].reshape([1,-1,1])

def full2mode(state, id, normalize=True):
    state = state[idx[id]].copy().reshape([1,-1,1])

    if normalize:
        if 'V' in states[id]:
            state[:,state2loc(id)['V']] = normalize_speed(state[:,state2loc(id)['V']])
        if 'h' in states[id]:
            state[:,state2loc(id)['h']] = normalize_height(state[:,state2loc(id)['h']])
    return state

def normalize(state, id):
    state                       = state.copy()
    if 'V' in states[id]:
        state[:,state2loc(id)['V']] = normalize_speed(state[:,state2loc(id)['V']])
    if 'h' in states[id]:
        state[:,state2loc(id)['h']] = normalize_height(state[:,state2loc(id)['h']])
    return state

def normalize_speed(speed):
    return (speed - CITATION_V_MIN)/(CITATION_V_MAX - CITATION_V_MIN)

def denormalize_speed(speed):
    return speed*(CITATION_V_MAX - CITATION_V_MIN)+ CITATION_V_MIN
    
def normalize_height(height):
    return (height - CITATION_H_MIN)/(CITATION_H_MAX - CITATION_H_MIN)

def denormalize_height(height):
    return height*(CITATION_H_MAX - CITATION_H_MIN) + CITATION_H_MIN

def denormalize(state, id):
    " Denormalize state: 'lon', 'lat', latlon, full "
    state = state.copy()
    if 'V' in states[id]:
        state[:,state2loc(id)['V']] = denormalize_speed(state[:,state2loc(id)['V']])
    if 'h' in states[id]:
        state[:,state2loc(id)['h']] = denormalize_height(state[:,state2loc(id)['h']])
    return state

#### Utility functions ####
def get_tracking_id(tracked_states, mode, exclude = None):
    " Translate the list of tracked states, mode and excluded states to an unique tracking list and ID " 
    
    # Starting counter
    util_id = 0

    # Determine mode: longitudinal, lateral or both
    if mode == LON:
        util_id += 100
    elif mode == LAT:
        util_id += 200
    elif mode == LATLON:
        util_id += 300
    
    # Check if tracked states is a string and convert to a list if not
    if isinstance(tracked_states, str):
        tracked_states = [tracked_states]
        
    # Check tracked states
    if 'p' in tracked_states:
        util_id += 1
    if 'q' in tracked_states:
        util_id += 2
    if 'r' in tracked_states:
        util_id += 4
    if 'beta' in tracked_states:
        util_id += 8
    
    # Check if states are to be excluded
    if exclude is not None:
        if 'V' in exclude:
            util_id += 16
        if 'h' in exclude:
            util_id += 32
    return util_id

def get_tracked_states(id):
    try:
        tracked_states    = track_states[id]

    except KeyError as e:
        msg = 'Combination of mode, tracked states and/or excluded state is not implemented yet.\n'
        msg += 'ID: ' + str(id) + '\n'
        msg += 'Error: ' + str(e) + '\n'
        raise NotImplementedError(msg)

    if id in [ID_Q_LON, ID_Q_LON_EX_V, ID_Q_LON_EX_H, ID_Q_LON_EX_VH, ID_Q_LATLON]:
        cmd = use_elevator()
    elif id in [ID_P_LATLON]:
        cmd = use_aileron()
    elif id in [ID_PQ_LATLON,ID_PQ_LATLON_EX_V, ID_PQ_LATLON_EX_H, ID_PQ_LATLON_EX_VH]:
        cmd = use_elevator_and_aileron()
    elif id in [ID_PQ_BETA_LATLON, ID_PQ_BETA_LATLON_EX_V, ID_PQ_BETA_LATLON_EX_H, ID_PQ_BETA_LATLON_EX_VH]:
        cmd = use_all_actuators()
    return tracked_states, cmd

def get_state_names(id):
    return states[id]

def get_citation_state_size(mode):
    if mode == LON:
        return 5
    elif mode == LAT:
        raise NotImplementedError
    elif mode == LATLON:
        return 9

def get_initial_state(id, flight_condition = 3):
    if flight_condition == 0:
        initial_state   = CITATION_INITIAL_STATE_FC0
    if flight_condition == 1:
        initial_state   = CITATION_INITIAL_STATE_FC1
    if flight_condition == 2:
        initial_state   = CITATION_INITIAL_STATE_FC2
    if flight_condition == 3:
        initial_state   = CITATION_INITIAL_STATE_FC3

    return initial_state[idx[id]].reshape([1,-1,1])

def display_tracking_info(id):
    if id - 200 < 0:
        mode = LON
    elif id - 300 < 0:
        mode = LAT
    elif id - 400 < 0:
        mode = LATLON
    track = ', '.join(np.array(states[id])[track_states[id]])
    info_msg = '\nTracking Info:\n  mode: ' + mode + '\n  tracks: ' + track
    print(info_msg + '\n')

def get_action_size(id):
    if id in [ID_Q_LON, ID_Q_LON_EX_V, ID_Q_LON_EX_H, ID_Q_LON_EX_VH, ID_P_LATLON, ID_Q_LATLON]:   
        return 1
    elif id in [ID_PQ_LATLON,ID_PQ_LATLON_EX_V, ID_PQ_LATLON_EX_H, ID_PQ_LATLON_EX_VH]:
        return 2 
    elif id in [ID_PQ_BETA_LATLON, ID_PQ_BETA_LATLON_EX_V, ID_PQ_BETA_LATLON_EX_H, ID_PQ_BETA_LATLON_EX_VH]:
        return 3
    else:
        raise ValueError("'tracked_states' does not correspond to a predefined action size.")

def get_state_error_weights(id):
    weights = np.ones(len(states[id]))
    if id in [ID_PQ_BETA_LATLON, ID_PQ_BETA_LATLON_EX_V, ID_PQ_BETA_LATLON_EX_H, ID_PQ_BETA_LATLON_EX_VH]:
        weights[state2loc(id)['q']] = 2
        weights[state2loc(id)['beta']] = 100
    return weights

def loc(state, id):
    return state2loc(id)[state]

def citloc(state):
    return state2citloc(ID_LATLON)[state]

def command(cmd, actions):
    raise NotImplementedError("Run tracked_states() first.")

def use_elevator():
    return lambda cmd, action: set_elevator(cmd, action.flatten())

def use_aileron():
    return lambda cmd, action: set_aileron(cmd, action.flatten())

def use_rudder():
    return lambda cmd, action: set_rudder(cmd, action.flatten())

def use_elevator_and_aileron():
    return lambda cmd, action: set_elevator(set_aileron(cmd, action.flatten()[1]), action.flatten()[0])

def use_all_actuators():
    return lambda cmd, action: set_elevator(set_aileron(set_rudder(cmd, action.flatten()[2]), action.flatten()[1]), action.flatten()[0])

def set_elevator(cmd, action):
    cmd[0] = action
    return cmd

def set_aileron(cmd, action):
    cmd[1] = action
    return cmd

def set_rudder(cmd, action):
    cmd[2] = action
    return cmd


# EXCITATION_FREQUENCIES = [(2.5, np.deg2rad(0.05)), (5, np.deg2rad(0.02)), (10, np.deg2rad(0.07)), (20, np.deg2rad(0.03)), (30, np.deg2rad(0.05))]
# def excitation_generator(action_shape, dt):
#     t = 0.0 
#     action_size = np.empty(action_shape).size
#     phase = np.deg2rad(np.random.randint(0, 360, [len(EXCITATION_FREQUENCIES), action_size]))

#     while True:
#         excitation = np.zeros(action_shape)
#         for i in range(len(EXCITATION_FREQUENCIES)):
#             excitation += EXCITATION_FREQUENCIES[i][1]*np.sin(2*np.pi*EXCITATION_FREQUENCIES[i][0]*(t + phase[None,i,:])) 
#         yield excitation
#         t += dt
