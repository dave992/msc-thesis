import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

import utils.phlab as phlab
from agents.base import BaseAgent

DEFAULT_HIDDEN_LAYER_SIZE = [50,50,50]
DEFAULT_LR_ACTOR = 0.05
DEFAULT_LR_CRITIC = 0.1
DEFAULT_GAMMA = 0.4
DEFAULT_USE_BIAS = False
DEFAULT_SPLIT = True
DEFAULT_TARGET_NETWORK = True
DEFAULT_TARGET_NETWORK_TAU = 0.001
DEFAULT_KERNEL_STDDEV = 0.1
DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_SESSION_CONFIG = None
DEFAULT_MODE_ID = phlab.ID_LATLON

class Agent(BaseAgent):
    "Heuristic Dynamic Programming based Reinforcement Learning agent"

    def __init__(self, load_path = None, **kwargs):
        # Hide info messages
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # Create default graph
        self.graph = tf.Graph()
            
        if load_path is None:   

            # Store parameters
            self.input_size = np.array(kwargs['input_size'], ndmin = 1)
            self.output_size = kwargs['output_size']
            self.hidden_layer_size = kwargs.get('hidden_layer_size', DEFAULT_HIDDEN_LAYER_SIZE)
            self.lr_critic = kwargs.get('lr_critic', DEFAULT_LR_CRITIC)
            self.lr_actor = kwargs.get('lr_actor', DEFAULT_LR_ACTOR)
            self.use_bias = kwargs.get('use_bias', DEFAULT_USE_BIAS)
            self.split = kwargs.get('split', DEFAULT_SPLIT)
            self.target_network = kwargs.get('target_network', DEFAULT_TARGET_NETWORK)
            self.tau = kwargs.get('tau', DEFAULT_TARGET_NETWORK_TAU)
            self.gamma = kwargs.get('gamma', DEFAULT_GAMMA)
            self.kernel_stddev = kwargs.get('kernel_stddev', DEFAULT_KERNEL_STDDEV)
            self.activation = kwargs.get('activation', DEFAULT_ACTIVATION)
            self.use_delta, self.tracked_states = kwargs.get('use_delta', (False, None))
            self.mode_id = kwargs.get('mode_id', DEFAULT_MODE_ID)
            session_config = kwargs.get('session_config', DEFAULT_SESSION_CONFIG)
            
            self.state_size = self.input_size[0]
            self.reference_size = self.input_size[1] if self.input_size.size == 2 else 0

            # Layer keyworded arguments
            self.hidden_layer_kwargs = {
                'kernel_initializer': tf.initializers.truncated_normal(stddev=self.kernel_stddev),
                'bias_initializer': tf.constant_initializer(.0),
                'use_bias': self.use_bias,
                'activation': self.activation
            }
            self.output_layer_kwargs = self.hidden_layer_kwargs.copy()
            self.output_layer_kwargs.pop('activation')

            # Save hyperparameters
            self.hyperparameters = pd.Series(kwargs)

            # Create tensorflow session
            self.session = tf.Session(graph = self.graph, config = session_config)

            # Build actor-critic and operations
            with self.graph.as_default(): # pylint: disable=E1129
                self.build_critic(self.lr_critic)
                if self.split:
                    self.build_split_actor(self.lr_actor)
                else:
                    self.build_actor(self.lr_actor)
                self.summary_op = tf.summary.merge_all()
                self.saver = tf.train.Saver(max_to_keep=20)
                self.session.run(tf.global_variables_initializer())
                

            # Switch beteen using target network and normal architecture
            if self.target_network:
                self.target_value_derivative = self.predict_tcritic
                self.session.run([tf.assign(self.vars_tcritic[i], self.vars_critic[i]) for i in range(len(self.vars_critic))])
            else:
                self.target_value_derivative = self.predict_critic
        else:
            raise NotImplementedError('Load not fully implemented!')
            self.load(load_path)
        self.trim = np.zeros(self.output_size)


    def build_critic(self, learn_rate):
        " Build the critic neural network "

        # Placeholders
        self.x_critic = tf.placeholder(tf.float32, shape=[None, self.state_size], name='state_input')
        self.x_gradient_critic = tf.placeholder(tf.float32, shape=[None, self.state_size], name='gradient_input')
        self.x_lr_critic = tf.placeholder_with_default(learn_rate, shape=[], name='learn_rate_input')
        if self.reference_size > 0:
            self.x_ref_critic = tf.placeholder(tf.float32, shape=[None, self.reference_size], name='reference_input')

        # Layer parameters
        h_kwargs = self.hidden_layer_kwargs.copy()
        y_kwargs = self.output_layer_kwargs

        # Shape input tensors based on preferences
        if self.reference_size > 0:
            if self.use_delta:
                x_masked = tf.boolean_mask(self.x_critic, self.tracked_states, axis=1)
                x_masked.set_shape([None, self.reference_size])
                x_ref = x_masked - self.x_ref_critic
            else:
                x_ref = self.x_ref_critic
            input_layer = tf.concat([self.x_critic, x_ref], axis=1)
        else:
            input_layer = self.x_critic
        x = input_layer

        # Add scope to the network
        with tf.variable_scope('critic'):
            
            # Build network
            for layer in range(len(self.hidden_layer_size)):
                x = tf.layers.dense(x, self.hidden_layer_size[layer], name = 'dense_'+ str(layer), **h_kwargs)              
            self.output_critic = tf.layers.dense(x, self.state_size, activation=None, name='critic_output', **y_kwargs)
            self.y_critic = self.output_critic
            
            # Get critic variables
            self.vars_critic = tf.trainable_variables(scope = 'critic')
            self.vars_critic_stacked = tf.concat([tf.reshape(self.vars_critic[i], [-1]) for i in range(len(self.vars_critic))], axis=0)
            

            # Update operations
            grads_and_vars = zip(tf.gradients(self.y_critic, self.vars_critic, self.x_gradient_critic), self.vars_critic)
            self.optimizer_critic = tf.train.GradientDescentOptimizer(self.x_lr_critic)
            self.optimize_critic = self.optimizer_critic.apply_gradients(grads_and_vars) 

        
        with tf.variable_scope('critic_target'):
            if self.target_network:
                x = input_layer
                # Build target network
                for layer in range(len(self.hidden_layer_size)):
                    x = tf.layers.dense(x, self.hidden_layer_size[layer], name = 'dense_'+ str(layer), **h_kwargs)              
                self.output_tcritic = tf.layers.dense(x, self.state_size, activation=None, name='critic_output', **y_kwargs)
                self.y_tcritic = self.output_tcritic

                # Get target critic variables
                self.vars_tcritic = tf.trainable_variables(scope = 'critic_target')

                # Update operations
                self.optimize_tcritic = [tf.assign(self.vars_tcritic[i], self.tau*self.vars_critic[i] + (1-self.tau)*self.vars_tcritic[i]) for i in range(len(self.vars_critic))]

                # Bookkeeping
                print('Critic & Target Critic Network build.')
            else:
                # Bookkeeping
                print('Critic Network build.')
        

    def build_split_actor(self, learn_rate):
        " Build the split longitudinal and lateral actor neural networks "
        
        ID_LON          = phlab.ID_LON
        ID_LAT          = phlab.ID_LAT
        LON_STATE_IDX   = [phlab.state2loc(self.mode_id)[n] for n in phlab.states[ID_LON]]
        LAT_STATE_IDX   = [phlab.state2loc(self.mode_id)[n] for n in phlab.states[ID_LAT]]

        # Layer parameters
        h_kwargs = self.hidden_layer_kwargs
        y_kwargs = self.output_layer_kwargs

        # Add scope to the network
        with tf.variable_scope('actor'):
            # General placeholder
            self.x_actor            = tf.placeholder(tf.float32, shape = [None, self.state_size], name = 'state_input')
            self.x_ref_actor        = tf.placeholder(tf.float32, shape = [None, self.reference_size], name='reference_input')
            self.x_gradient_actor   = tf.placeholder(tf.float32, shape = [None, self.output_size], name = 'gradient_input') 
            self.x_lr_actor         = tf.placeholder_with_default(learn_rate, shape=[], name='learn_rate_input')

            n = 0 
            x_ref_lon, x_ref_lat = [], []
            for i, track in enumerate(self.tracked_states):
                if track:
                    if i in LON_STATE_IDX:
                        x_ref_lon.append(self.x_ref_actor[:,n])
                    if i in LAT_STATE_IDX:
                        x_ref_lat.append(self.x_ref_actor[:,n])
                    n += 1

            with tf.variable_scope('longitudinal'):
                # Placeholders
                x = []
                for n in LON_STATE_IDX:
                    x.append(self.x_actor[:,n])
                self.x_lon_actor        = tf.stack(x, axis=1)
                self.x_ref_lon_actor    = tf.stack(x_ref_lon, axis=1)

                # Shape input tensors based on preferences
                if self.use_delta:
                    lon_tracked_states = [self.tracked_states[n] for n in LON_STATE_IDX]
                    x_masked = tf.boolean_mask(self.x_lon_actor, lon_tracked_states, axis=1)
                    x_masked.set_shape([None, np.sum(lon_tracked_states)])
                    x_ref_lon = x_masked - self.x_ref_lon_actor
                else: 
                    x_ref_lon = self.x_ref_lon_actor

                x_lon = tf.concat([self.x_lon_actor, x_ref_lon], axis=1)
                
                # Build network
                for layer in range(len(self.hidden_layer_size)):
                    name_str = 'dense_' + str(layer)
                    x_lon = tf.layers.dense(x_lon, self.hidden_layer_size[layer], name=name_str, **h_kwargs)

            with tf.variable_scope('lateral'):
                # Placeholders
                x = []
                for n in LAT_STATE_IDX:
                    x.append(self.x_actor[:,n])
                self.x_lat_actor        = tf.stack(x, axis=1)
                self.x_ref_lat_actor    = tf.stack(x_ref_lat, axis=1)

                # Shape input tensors based on preferences
                if self.use_delta:
                    lat_tracked_states = [self.tracked_states[n] for n in LAT_STATE_IDX]
                    x_masked = tf.boolean_mask(self.x_lat_actor, lat_tracked_states, axis=1)
                    x_masked.set_shape([None, np.sum(lat_tracked_states)])
                    x_ref_lat = x_masked - self.x_ref_lat_actor
                else: 
                    x_ref_lat = self.x_ref_lat_actor

                x_lat = tf.concat([self.x_lat_actor, x_ref_lat], axis=1)

                # Build network
                for layer in range(len(self.hidden_layer_size)):
                    name_str = 'dense_' + str(layer)
                    x_lat = tf.layers.dense(x_lat, self.hidden_layer_size[layer], name=name_str, **h_kwargs)

            self.output_actor = [tf.layers.dense(x_lon, 1, activation=None, name='actor_output_0', **y_kwargs)]
            for action in range(self.output_size - 1):
                self.output_actor.append(tf.layers.dense(x_lat, 1, activation=None, name='actor_output_'+str(action+1), **y_kwargs))
            self.y_actor = tf.concat(self.output_actor, axis=1) 
            
            # Bookkeeping
            print('Actor network build.')
            self.vars_actor = tf.trainable_variables(scope = 'actor')
            self.vars_actor_stacked = tf.concat([tf.reshape(self.vars_actor[i], [-1]) for i in range(len(self.vars_actor))],axis=0)
            
            # Update operations
            grads_and_vars = zip(tf.gradients(self.y_actor, self.vars_actor, self.x_gradient_actor), self.vars_actor)
            self.optimizer_actor = tf.train.GradientDescentOptimizer(self.x_lr_actor)
            self.optimize_actor = self.optimizer_actor.apply_gradients(grads_and_vars)

            # Gradient through actor
            with tf.variable_scope("dactiondx"):
                gradients_actor = []
                for action in self.output_actor:
                    gradients_actor += tf.gradients(action, self.x_actor)
                self.gradient_actor_op = tf.stack(gradients_actor, axis=1)

    def build_actor(self, learn_rate):
        " Build the actor neural network "

        # Add scope to the network
        with tf.variable_scope('actor'):
            # Placeholders
            self.x_actor = tf.placeholder(tf.float32, shape = [None, self.state_size], name = 'state_input')
            self.x_gradient_actor = tf.placeholder(tf.float32, shape = [None, self.output_size], name = 'gradient_input')
            self.x_lr_actor = tf.placeholder_with_default(learn_rate, shape=[], name='learn_rate_input')
            
            if self.reference_size > 0:
                self.x_ref_actor = tf.placeholder(tf.float32, shape=[None, self.reference_size], name='reference_input')

            # Layer parameters
            h_kwargs = self.hidden_layer_kwargs
            y_kwargs = self.output_layer_kwargs

            # Shape input tensors based on preferences
            if self.reference_size > 0:
                if self.use_delta:
                    x_masked = tf.boolean_mask(self.x_actor, self.tracked_states, axis=1)
                    x_masked.set_shape([None, self.reference_size])
                    x_ref = x_masked - self.x_ref_actor
                else:
                    x_ref = self.x_ref_actor

                x = tf.concat([self.x_actor, x_ref], axis=1)
            else:
                x = self.x_actor
            
            # Build network
            for layer in range(len(self.hidden_layer_size)):
                name_str = 'dense_' + str(layer)
                x = tf.layers.dense(x, self.hidden_layer_size[layer], name=name_str, **h_kwargs)
            
            self.output_actor = []
            for action in range(self.output_size):
                self.output_actor.append(tf.layers.dense(x, 1, activation=None, name='actor_output_'+str(action), **y_kwargs))
            self.y_actor = tf.concat(self.output_actor, axis=1) 
            
            # Bookkeeping
            print('Actor network build.')
            self.vars_actor = tf.trainable_variables(scope = 'actor')
            self.vars_actor_stacked = tf.concat([tf.reshape(self.vars_actor[i], [-1]) for i in range(len(self.vars_actor))],axis=0)
            
            # Update operations
            grads_and_vars = zip(tf.gradients(self.y_actor, self.vars_actor, self.x_gradient_actor), self.vars_actor)
            self.optimizer_actor = tf.train.GradientDescentOptimizer(self.x_lr_actor)
            self.optimize_actor = self.optimizer_actor.apply_gradients(grads_and_vars)

            # Gradient through actor
            with tf.variable_scope("dactiondx"):
                gradients_actor = []
                for action in self.output_actor:
                    gradients_actor += tf.gradients(action, self.x_actor)
                self.gradient_actor_op = tf.stack(gradients_actor, axis=1)

    def create_tensorboard_writer(self, file):
        self.writer = tf.summary.FileWriter(file, self.session.graph)
    
    def predict_critic(self, state, reference = None):
        # Predict critic 
        feed = {self.x_critic: np.squeeze(state, axis=2)}
        if reference is not None:
            feed[self.x_ref_critic] = np.squeeze(reference, axis=2)
        lmbda = self.session.run(self.y_critic, feed_dict=feed)
        lmbda = np.expand_dims(lmbda, axis=1)
        return lmbda
    value_derivative = predict_critic

    def predict_tcritic(self, state, reference = None):
        # Predict critic 
        feed = {self.x_critic: np.squeeze(state, axis=2)}
        if reference is not None:
            feed[self.x_ref_critic] = np.squeeze(reference, axis=2)
        lmbda = self.session.run(self.y_tcritic, feed_dict=feed)
        lmbda = np.expand_dims(lmbda, axis=1)
        return lmbda
    
    
    def update_critic(self, state, gradient, reference = None, learn_rate = None):
        # Create update feed dict
        feed = {self.x_critic: np.squeeze(state, axis=2),
                self.x_gradient_critic: np.squeeze(gradient, axis=1)}
        if reference is not None:
            feed[self.x_ref_critic] = np.squeeze(reference, axis=2)
        if learn_rate is not None:
            feed[self.x_lr_critic] = learn_rate

        # Update Critic
        if self.target_network:
            self.session.run([self.optimize_critic, self.optimize_tcritic], feed_dict=feed)
        else:
            self.session.run(self.optimize_critic, feed_dict=feed)

    def gradient_actor(self, state, reference=None):
        # Calculate gradient
        feed = {self.x_actor: np.squeeze(state, axis=2)}
        if reference is not None:
            feed[self.x_ref_actor] = np.squeeze(reference, axis=2)
        return self.session.run(self.gradient_actor_op, feed_dict = feed)

    def predict_actor(self, state, reference = None):
        # Predict action
        feed = {self.x_actor: np.squeeze(state, axis=2)}
        if reference is not None:
            feed[self.x_ref_actor] = np.squeeze(reference, axis=2)
        actions = self.session.run(self.y_actor, feed_dict=feed) + self.trim

        return actions
    action = predict_actor
    
    def update_actor(self, state, gradient, reference = None, learn_rate = None):
        # Create update feed dict
        feed = {self.x_actor: np.squeeze(state, axis=2),
                self.x_gradient_actor: np.squeeze(gradient, axis=1)}
        if reference is not None:
            feed[self.x_ref_actor] = np.squeeze(reference, axis=2)
        if learn_rate is not None:
            feed[self.x_lr_actor] = learn_rate

        # Update Actor
        self.session.run(self.optimize_actor, feed_dict=feed)

    def mean_and_variance_critic(self):
        return self.session.run(tf.nn.moments(self.vars_critic_stacked, axes=[0]))

    def mean_and_variance_actor(self):
        return self.session.run(tf.nn.moments(self.vars_actor_stacked, axes=[0]))

    def assign_actor(self, weights):
        assign_ops = []
        for weight, var in zip(weights, self.vars_actor):
            assign_ops.append(tf.assign(var, weight))
        self.session.run(assign_ops)

    def assign_critic(self, weights):
        assign_ops = []
        for weight, var in zip(weights, self.vars_critic):
            assign_ops.append(tf.assign(var, weight))
        self.session.run(assign_ops)

    def load(self, file_name):
        # Load hyper-parameters
        self.hyperparameters = pd.read_pickle(os.path.splitext(file_name)[0] + '.pkl')
        self.input_size = np.array(self.hyperparameters['input_size'], ndmin = 1)
        self.output_size = self.hyperparameters['output_size']
        self.hidden_layer_size = self.hyperparameters['hidden_layer_size']
        self.lr_critic = self.hyperparameters['lr_critic']
        self.lr_actor = self.hyperparameters['lr_actor']
        self.gamma = self.hyperparameters['gamma']
        self.activation = self.hyperparameters['activation']
        self.use_delta, self.tracked_states = self.hyperparameters['use_delta']
        session_config = self.hyperparameters['session_config']

        self.state_size = self.input_size[0]
        self.reference_size = self.input_size[1] if self.input_size.size == 2 else 0

        # Create tensorflow session
        self.session = tf.Session(graph = self.graph, config = session_config)

        # Build actor-critic and operations
        with self.graph.as_default(): # pylint: disable=E1129
            self.build_actor(self.lr_actor)
            self.build_critic(self.lr_critic)
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=20)
            self.saver.restore(self.session, file_name)
    
    def save(self, file_path, global_step = None):
        # Save Hyper-parameters
        if global_step is None or global_step == 0:
            pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True) 
            self.hyperparameters.to_pickle(os.path.splitext(file_path)[0] + '.pkl')
        
        # Save checkpoint
        self.saver.save(self.session, file_path, global_step=global_step)

        
