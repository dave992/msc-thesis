class BaseAgent():
    """ 
    Base class for different agents. All agents should containt the following function implementations:
        reset() : Reset the environment and all agent parameters for starting a new episode.
        step()  : Step the environment forward with one timestep. 
        render(): OpenAI compat function, renders the environment at the current timestep. 
    """
    
    def __init__(self, action_space, state_space):
        " Initialize the BaseAgent class with all environment info needed. "
        self.action_space = action_space
        self.state_space = state_space
    
    def reset(self):
        raise NotImplementedError
        
    def step(self, action = None):
        raise NotImplementedError
        
    def render(self):
        raise NotImplementedError
        