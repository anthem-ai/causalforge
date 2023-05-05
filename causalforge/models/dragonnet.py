"""
Code Skeleton from: https://github.com/uber/causalml/blob/master/causalml/inference/tf/dragonnet.py
Original implementation: https://github.com/claudiashi57/dragonnet 
"""

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
import tensorflow as tf
import keras.backend as K

from .utils import (
    dragonnet_loss_binarycross,
    EpsilonLayer,
    regression_loss,
    binary_classification_loss,
    treatment_accuracy,
    track_epsilon,
    make_tarreg_loss,
)

def convert_pd_to_np(*args):
    output = [obj.to_numpy() if hasattr(obj, "to_numpy") else obj for obj in args]
    return output if len(output) > 1 else output[0]

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
    
    def build(self,user_params):
        
        if 'input_dim' not in user_params:
            raise Exception("input_dim must be specified!")
        
        params = {
            'neurons_per_layer': 200, 
            'reg_l2': 0.01, 
            'targeted_reg': True, 
            'verbose': True, 
            'val_split': 0.2, 
            'ratio': 1.0, 
            'batch_size': 64,
            'epochs': 100,
            'learning_rate': 1e-5, 
            'momentum': 0.9,
            'use_adam': True,
            'adam_epochs':30, 
            'adam_learning_rate': 1e-3
        }
        
        for k in params:
            if k in user_params:
                params[k] = user_params[k]
                
        params['input_dim' ] = user_params['input_dim']
       
        # 
        K.clear_session()
        
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
        self.params = params
    
    def support_ite(self):
        return True 
    
    def predict_ite(self, X):
        preds = self.model.predict(X)
        return (preds[:, 1] - preds[:, 0])
    
    def predict_ate(self, X):
        return np.mean(self.predict_ite(X))
    
    def fit(self, X, treatment, y):
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        y = np.hstack((y.reshape(-1, 1), treatment.reshape(-1, 1)))
        metrics = [
            regression_loss,
            binary_classification_loss,
            treatment_accuracy,
            track_epsilon,
        ]
        
        if 'targeted_reg' in self.params and self.params['targeted_reg']:
            loss = make_tarreg_loss(ratio=self.params['ratio'], dragonnet_loss=dragonnet_loss_binarycross)
        else:
            loss = dragonnet_loss_binarycross
        
        if 'use_adam' in self.params and self.params['use_adam']: 
            self.model.compile(
                optimizer=Adam(lr=self.params['adam_learning_rate']), loss=loss, metrics=metrics
            )

            adam_callbacks = [
                TerminateOnNaN(),
                EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0),
                ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.5,
                    patience=5,
                    verbose=self.params['verbose'],
                    mode="auto",
                    min_delta=1e-8,
                    cooldown=0,
                    min_lr=0,
                ),
            ]
            
            self.model.fit(
                X,
                y,
                callbacks=adam_callbacks,
                validation_split=self.params['val_split'],
                epochs=self.params['adam_epochs'],
                batch_size=self.params['batch_size'],
                verbose=self.params['verbose']
            )
        
        # 
        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=40, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=self.params['verbose'],
                mode="auto",
                min_delta=0.0,
                cooldown=0,
                min_lr=0,
            ),
        ]
        self.model.compile(
            optimizer=SGD(lr=self.params['learning_rate'], momentum=self.params['momentum'], 
                          nesterov=True),
            loss=loss,
            metrics=metrics,
        )
        self.model.fit(
            X,
            y,
            callbacks=sgd_callbacks,
            validation_split=self.params['val_split'],
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=self.params['verbose']
        )
    
    
    
