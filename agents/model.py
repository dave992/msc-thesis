from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.python.ops.parallel_for import gradients

DEFAULT_HIDDEN_LAYER_SIZE   = [6, 6]
DEFAULT_LR                  = 1.0
DEFAULT_ACTIVATION          = tf.nn.tanh
DEFAULT_SESSION_CONFIG      = None
DEFAULT_INCREMENTAL         = False

class NeuralNetwork():
    
    def __init__(self, load = None, **kwargs):
        
        # Create default graph
        self.graph = tf.Graph()

        # Read kwargs
        self.state_size         = kwargs['state_size']
        self.action_size        = kwargs['action_size']
        self.hidden_layer_size  = kwargs.get('hidden_layer_size', DEFAULT_HIDDEN_LAYER_SIZE)
        self.lr                 = kwargs.get('lr', DEFAULT_LR)
        self.activation         = kwargs.get('activation', DEFAULT_ACTIVATION)
        self.predict_delta      = kwargs.get('predict_delta', DEFAULT_INCREMENTAL)
        self.weights            = kwargs.get('weights', np.ones(self.state_size)).reshape([1, -1])
        session_config          = kwargs.get('session_config', DEFAULT_SESSION_CONFIG)
    

        # Save hyperparameters
        self.hyperparameters = pd.Series(kwargs)

        # Create tensorflow session
        self.session = tf.Session(graph = self.graph, config = session_config)

        # Build actor-critic and operations
        with self.graph.as_default(): # pylint: disable=E1129
            self.build(self.lr)
            self.saver = tf.train.Saver(max_to_keep=20)
            if load is not None:
                self.saver.restore(self.session, load)
                print('Aircraft model restored from:', load)
            else:
                self.session.run(tf.global_variables_initializer())

    def build(self, learn_rate):

        # Add scope to the network
        with tf.variable_scope('model'):
            # Placeholders
            self.x_state    = tf.placeholder(tf.float32, shape=[None, self.state_size], name = 'state_input')
            self.x_action   = tf.placeholder(tf.float32, shape=[None,self.action_size], name='action_input')
            self.x_lr       = tf.placeholder_with_default(learn_rate, shape=[], name='learn_rate_input')
            self.y_label    = tf.placeholder(tf.float32, shape=[None, self.state_size], name='label')

            # Network 
            x = tf.concat([self.x_state, self.x_action], axis=1, name='model_input')
            for layer in range(len(self.hidden_layer_size)):
                x = tf.layers.dense(x, self.hidden_layer_size[layer], activation = self.activation, name = 'dense_'+ str(layer))
                with tf.variable_scope('dense_' + str(layer), reuse = True):
                    tf.summary.histogram('kernel', tf.get_variable('kernel'))
                    tf.summary.histogram('bias', tf.get_variable('bias'))
            self.output_model = []
            for y in range(self.state_size):                 
                self.output_model.append(tf.layers.dense(x, 1, activation = None, name = 'model_output_'+str(y)))
            self.y_state = tf.concat(self.output_model, axis=1)

        # Update Rules
        self.loss = tf.losses.mean_squared_error(labels = self.y_label, predictions = self.y_state, weights=self.weights)
        self.optimize = tf.train.AdamOptimizer(self.x_lr).minimize(self.loss)

        # Gradients
        with tf.variable_scope("dxdaction"):
                grads = []
                for y in self.output_model:
                    grads += tf.gradients(y, self.x_action)
                self.gradient_action_op = tf.stack(grads, axis=1)
        with tf.variable_scope("dxdx"):
                grads = []
                for y in self.output_model:
                    grads += tf.gradients(y, self.x_state)
                self.gradient_state_op = tf.stack(grads, axis=1)


    def update(self, state, action, next_state, learn_rate = None):
        # Reshape
        state       = np.reshape(state, [-1, self.state_size])
        action      = np.reshape(action, [-1, self.action_size])
        next_state  = np.reshape(next_state, [-1, self.state_size])
        
        # Calculate increment
        if self.predict_delta:
            next_state = next_state - state

        # Update
        feed = {self.x_state: state,
                self.x_action: action, 
                self.y_label: next_state}
        if learn_rate is not None:
            feed[self.x_lr] = learn_rate
        return self.session.run([self.loss, self.optimize], feed_dict=feed)

    def calculate_loss(self, state, action, next_state):
        # Reshape
        state       = np.reshape(state, [-1, self.state_size])
        action      = np.reshape(action, [-1, self.action_size])
        next_state  = np.reshape(next_state, [-1, self.state_size])
        
        # Calculate increment
        if self.predict_delta:
            next_state = next_state - state

        # Update
        feed = {self.x_state: state,
                self.x_action: action, 
                self.y_label: next_state}
        return self.session.run(self.loss, feed_dict=feed)

    def predict(self, state, action):
        # Reshape 
        state       = np.reshape(state, [-1, self.state_size])
        action      = np.reshape(action, [-1, self.action_size])

        # Calculate prediction
        feed = {self.x_state: state, 
                self.x_action: action}
        return self.session.run(self.y_state, feed_dict=feed)

    def gradient_state(self, state, action):
        # Reshape
        state = np.reshape(state, [-1, self.state_size])
        action = np.reshape(action, [-1, self.action_size])

        # Calculate gradient
        feed        = {self.x_state: state, self.x_action: action}
        gradient    = self.session.run(self.gradient_state_op, feed_dict=feed)
        
        if self.predict_delta:
            gradient += 0#np.eye(self.state_size)
        return gradient

    def gradient_action(self, state, action):
        # Reshape
        state = np.reshape(state, [-1, self.state_size])
        action = np.reshape(action, [-1, self.action_size])

        # Calculate gradient
        feed        = {self.x_state: state, self.x_action: action}
        return self.session.run(self.gradient_action_op, feed_dict=feed)

DEFAULT_LS_PREDICT_DELTA = True
DEFAULT_LS_BUFFER_LENGTH = 10

class LeastSquares():

    def __init__(self, **kwargs):

        # Read kwargs
        self.state_size     = kwargs['state_size']
        self.action_size    = kwargs['action_size']
        self.predict_delta  = kwargs.get('predict_delta', DEFAULT_LS_PREDICT_DELTA)
        self.buffer_length  = kwargs.get('buffer_length', DEFAULT_LS_BUFFER_LENGTH)

        # Create buffers for data and parameter estimation
        self.buffer_size = 0
        self.buffer_x = np.zeros([self.buffer_length, self.state_size])
        self.buffer_u = np.zeros([self.buffer_length, self.action_size])
        self.buffer_1 = np.ones([self.buffer_length, 1])
        self.buffer_y = np.zeros([self.buffer_length, self.state_size])
        if self.predict_delta:
            self.W = np.zeros([self.state_size, self.state_size + self.action_size + 1])
        else:
            self.W = np.eye(self.state_size, self.state_size + self.action_size + 1)

    def update(self, state, action, next_state):
        # Reshape
        state       = np.reshape(state, [-1, self.state_size])
        action      = np.reshape(action, [-1, self.action_size])
        next_state  = np.reshape(next_state, [-1, self.state_size])

        # Roll and replace buffers
        self.buffer_x = np.roll(self.buffer_x, 1, axis=0)
        self.buffer_x[0:] = state
        self.buffer_u = np.roll(self.buffer_u, 1, axis=0)
        self.buffer_u[0:] = action
        self.buffer_y = np.roll(self.buffer_y, 1, axis=0)
        self.buffer_y[0:] = next_state
        self.buffer_size += 1

        if self.buffer_size == self.buffer_length:
            y = self.buffer_y - self.buffer_x if self.predict_delta else self.buffer_y
            X = np.hstack([self.buffer_x, self.buffer_u, self.buffer_1])
        W, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.W = W.T

    def predict(self, state, action):
        # Reshape to remove possible third dimension
        state = np.reshape(state, [-1, self.state_size])
        action = np.reshape(action, [-1, self.action_size])
        
        x = np.hstack([state, action, 1]).T
        return np.matmul(self.W, x).T[:,:,None]

    def gradient_state(self, state=None, action=None):
        gradients = self.W[:,:self.state_size]
        if self.predict_delta:
            gradients += np.identity(self.state_size)
        return gradients

    def gradient_action(self, state=None, action=None):
        gradients = self.W[:,self.state_size:]
        return gradients 

DEFAULT_RLS_GAMMA       = 0.8
DEFAULT_RLS_COVARIANCE  = 1
DEFAULT_RLS_CONSTANT    = True

class RecursiveLeastSquares():

    def __init__(self, **kwargs):
        
        # Read kwargs
        self.state_size         = kwargs['state_size']
        self.action_size        = kwargs['action_size']
        self.predict_delta      = kwargs.get('predict_delta', DEFAULT_INCREMENTAL)
        self.gamma              = kwargs.get('gamma', DEFAULT_RLS_GAMMA)
        self.covariance         = kwargs.get('covariance', DEFAULT_RLS_COVARIANCE)
        self.constant           = kwargs.get('constant', DEFAULT_RLS_CONSTANT)
        self.nb_vars            = self.state_size + self.action_size
        if self.constant:
            self.nb_coefficients    = self.state_size + self.action_size + 1 
            self.constant_array     = np.array([[[1]]]) 
        else: 
            self.nb_coefficients = self.state_size + self.action_size

        # Initialize 
        self.reset()

    def update(self, state, action, next_state):
        " Update parameters and covariance. "

        state       = state.reshape([-1,1])
        action      = action.reshape([-1,1])
        next_state  = next_state.reshape([-1,1])

        if self.skip_update:
            # Store x and u
            self.x              = state
            self.u              = action
            self.skip_update    = False
            return
        
        # Incremental switch
        if self.predict_delta:
            self.X[:self.state_size]                = state - self.x 
            self.X[self.state_size:self.nb_vars]    = action - self.u
            Y                                       = next_state - state
            # Store x and u
            self.x = state
            self.u = action
        else:
            self.X[:self.state_size]                = state
            self.X[self.state_size:self.nb_vars]    = action
            Y                                       = next_state

        # Error
        Y_hat   = np.matmul(self.X.T, self.W).T
        error   = (Y - Y_hat).T

        # Intermidiate computations
        covX        = np.matmul(self.cov, self.X)
        Xcov        = np.matmul(self.X.T, self.cov)
        gamma_XcovX = self.gamma + np.matmul(Xcov, self.X)

        # Update weights and covariance 
        self.W      = self.W + np.matmul(covX, error) / gamma_XcovX
        self.cov    = (self.cov - np.matmul(covX, Xcov) / gamma_XcovX) / self.gamma

    def predict(self, state, action):
        state = state.reshape([-1,1])
        action = action.reshape([-1,1])

        if self.predict_delta:
            self.X[:self.state_size]                = state - self.x
            self.X[self.state_size:self.nb_vars]    = action - self.u
            X_next_pred = state + np.matmul(self.W.T, self.X)
        else:
            self.X[:self.state_size]                = state
            self.X[self.state_size:self.nb_vars]    = action
            X_next_pred = np.matmul(self.W.T, self.X)
        return X_next_pred

    def gradient_state(self, state, action):
        gradients = self.W[:self.state_size,:].T
        if self.predict_delta:
            gradients = gradients #+ np.identity(self.state_size)
        return gradients

    def gradient_action(self, state, action):
        gradient = self.W[self.state_size:self.nb_vars,:].T
        return gradient

    def reset(self):
        " Reset parameters and covariance. Check if last state is  "

        self.X      = np.ones([self.nb_coefficients, 1])
        # self.W      = np.eye(self.nb_coefficients, self.state_size)
        self.W      = np.zeros([self.nb_coefficients, self.state_size])
        self.cov    = np.identity(self.nb_coefficients)*self.covariance
        
        if self.predict_delta:
            self.x = np.zeros([self.state_size, 1])
            self.u = np.zeros([self.action_size, 1])
            self.skip_update = True
        else: 
            self.skip_update = False

    def reset_covariance(self):
        self.cov = np.identity(self.nb_coefficients)*self.covariance
            
class rls():
    "Implementation of the Recursive Least Squares (RLS) filter as incremental model"

    def __init__(self, state_size, control_size, gamma=.8):
        "Populates attributes and initializes state covariance matrix and parameter matrix"

        # populate attributes
        self.gamma = gamma
        self.ss = state_size
        self.cs = control_size

        # initialize covariance and parameter matrix
        self.theta = np.zeros((self.ss+self.cs, self.ss))
        self.Cov = np.identity(self.ss+self.cs)

    def update(self, delta_x, delta_u, delta_x_next):
        "Updates RLS parameters based on one sample pair"
        
        # stack input data, predict, compute innovation
        self.X = np.vstack((delta_x, delta_u))
        self.delta_x_hat = np.matmul(self.X.T, self.theta).T
        self.epsilon = delta_x_next.T - self.delta_x_hat.T

        # intermidiate computations
        CX = np.matmul(self.Cov, self.X)
        XC = np.matmul(self.X.T, self.Cov)

        # update parameter and covariance matrix
        self.theta = self.theta + np.matmul(CX, self.epsilon) / (self.gamma+np.matmul(XC, self.X))
        self.Cov = (self.Cov - np.matmul(CX, XC) / (self.gamma + np.matmul(XC, self.X))) / self.gamma
        return

    def predict(self, delta_x, delta_u, x):
        "Predict next state using RLS parameters"

        # extract state and input matrix estimates
        F = self.theta[:self.ss,:].T
        G = self.theta[self.ss:,:].T
        
        # return state prediction  
        return x + np.matmul(F, delta_x) + np.matmul(G, delta_u)

    def get_grads(self):
        "Returns current estimates of state and input matrix"

        # extract state and input matrix estimates
        F = self.theta[:self.ss,:].T
        G = self.theta[self.ss:,:].T

        return F, G

    def reset(self):
        "Resets the covariance and parameter matrix"

        # reset covariance and parameter matrix
        self.theta = np.zeros((self.ss+self.cs, self.ss))
        self.Cov = np.identity(self.ss+self.cs)
