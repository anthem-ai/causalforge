#from ..model import Model

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
import tensorflow as tf


class EpsilonLayer(Layer):
    """
    Custom keras layer to allow epsilon to be learned during training process.
    """

    def __init__(self):
        """
        Inherits keras' Layer object.
        """
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        """
        Creates a trainable weight variable for this layer.
        """
        self.epsilon = self.add_weight(
            name="epsilon", shape=[1, 1], initializer="RandomNormal", trainable=True
        )
        super(EpsilonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]

class DragonNet(Model):
    
    def build(self,params):
        inputs = Input(shape=(params['input_dim'],), name="input")
        
        # representation
        x = Dense(
            units=params['neurons_per_layer'],
            activation="elu",
            kernel_initializer="RandomNormal",
        )(inputs)
        x = Dense(
            units=params['neurons_per_layer'],
            activation="elu",
            kernel_initializer="RandomNormal",
        )(x)
        x = Dense(
            units=params['neurons_per_layer'],
            activation="elu",
            kernel_initializer="RandomNormal",
        )(x)

        t_predictions = Dense(units=1, activation="sigmoid")(x)

        # HYPOTHESIS
        y0_hidden = Dense(
            units=int(params['neurons_per_layer'] / 2),
            activation="elu",
            kernel_regularizer=l2(params['reg_l2']),
        )(x)
        y1_hidden = Dense(
            units=int(params['neurons_per_layer'] / 2),
            activation="elu",
            kernel_regularizer=l2(params['reg_l2']),
        )(x)

        # second layer
        y0_hidden = Dense(
            units=int(params['neurons_per_layer'] / 2),
            activation="elu",
            kernel_regularizer=l2(params['reg_l2']),
        )(y0_hidden)
        y1_hidden = Dense(
            units=int(params['neurons_per_layer'] / 2),
            activation="elu",
            kernel_regularizer=l2(params['reg_l2']),
        )(y1_hidden)

        # third
        y0_predictions = Dense(
            units=1,
            activation=None,
            kernel_regularizer=l2(params['reg_l2']),
            name="y0_predictions",
        )(y0_hidden)
        y1_predictions = Dense(
            units=1,
            activation=None,
            kernel_regularizer=l2(params['reg_l2']),
            name="y1_predictions",
        )(y1_hidden)

        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name="epsilon")
        concat_pred = Concatenate(1)(
            [y0_predictions, y1_predictions, t_predictions, epsilons]
        )
        
        self.model = Model(inputs=inputs, outputs=concat_pred)

    
    
    