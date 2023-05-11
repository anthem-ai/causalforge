import time 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Concatenate
from keras.models import Model as keras_Model
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from causalforge.model import Model
from .utils import (
    EpsilonLayer,
    convert_pd_to_np
)


class BCAUSS(Model):
    
    def build(self,user_params):
        if 'input_dim' not in user_params:
            raise Exception("input_dim must be specified!")
        
        params = {
            'neurons_per_layer': 200, 
            "act_fn": "relu",
            'reg_l2': 0.01, 
            'verbose': True, 
            'val_split': 0.22, 
            'ratio': 1.0, 
            'optim': "sgd",
            'batch_size': 64,
            'epochs': 500,
            'learning_rate': 1e-5, 
            'momentum': 0.9,
            "use_bce": False, 
            "norm_bal_term": True, 
            "use_targ_term": False, 
            "ratio": 1., 
            "b_ratio": 1.,
            "bs_ratio": 1.,
            "scale_preds": True
        }
        
        for k in params:
            params[k] = user_params.get(k,params[k])
                
        params['input_dim' ] = user_params['input_dim']
       
        # 
        K.clear_session()
        
        inputs = Input(shape=(params['input_dim'],), name="input")
        
        # representation
        x = Dense(units=params['neurons_per_layer'], activation=params['act_fn'], kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=params['neurons_per_layer'], activation=params['act_fn'], kernel_initializer='RandomNormal')(x)
        x = Dense(units=params['neurons_per_layer'], activation=params['act_fn'], kernel_initializer='RandomNormal')(x)


        t_predictions = Dense(units=1, activation='sigmoid')(x)

        # HYPOTHESIS
        y0_hidden = Dense(units=int(params['neurons_per_layer'] / 2), activation=params['act_fn'], kernel_regularizer=regularizers.l2(params['reg_l2']))(x)
        y1_hidden = Dense(units=int(params['neurons_per_layer'] / 2), activation=params['act_fn'], kernel_regularizer=regularizers.l2(params['reg_l2']))(x)

        # second layer
        y0_hidden = Dense(units=int(params['neurons_per_layer'] / 2), activation=params['act_fn'], kernel_regularizer=regularizers.l2(params['reg_l2']))(y0_hidden)
        y1_hidden = Dense(units=int(params['neurons_per_layer'] / 2), activation=params['act_fn'], kernel_regularizer=regularizers.l2(params['reg_l2']))(y1_hidden)

        # third
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(params['reg_l2']), name='y0_predictions')(
            y0_hidden)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(params['reg_l2']), name='y1_predictions')(
            y1_hidden)

        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon')
        # logging.info(epsilons)
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
        
        # Additional 'inputs' for the labels
        y_true = Input(shape=(1,),name='y_true')
        t_true = Input(shape=(1,),name='t_true')
        
        model = keras_Model(inputs=[inputs,y_true,t_true], outputs=concat_pred)
        
        #  _                 
        # | |    ___  ___ ___
        # | |__ / _ \(_-<(_-<
        # |____|\___//__//__/
        
        ## binary_classification_loss
        t_pred = (t_predictions + 0.001) / 1.002
        binary_classification_loss = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

        ## regression_loss
        loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
        loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions)) 

        regression_loss = loss0 + loss1
        
        if params['use_bce']:
            vanilla_loss = regression_loss + binary_classification_loss 
        else:
            vanilla_loss = regression_loss 

        y_pred = t_true * y1_predictions + (1 - t_true) * y0_predictions

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        
        if params['use_targ_term']:
            y_pert = y_pred + epsilons * h
            targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))
        
        ## auto-balancing self-supervised objective
        ones_to_sum = K.repeat_elements(t_true / t_pred, rep=params['input_dim' ], axis=1)*inputs
        zeros_to_sum = K.repeat_elements((1 - t_true) / (1 - t_pred), rep=params['input_dim' ], axis=1)*inputs
        
        if params['norm_bal_term']:
            ones_mean = tf.math.reduce_sum(ones_to_sum,0)/tf.math.reduce_sum(t_true / t_pred,0)
            zeros_mean = tf.math.reduce_sum(zeros_to_sum,0)/tf.math.reduce_sum((1 - t_true) / (1 - t_pred),0)
        else:
            ones_mean = tf.math.reduce_sum(ones_to_sum,0)
            zeros_mean = tf.math.reduce_sum(zeros_to_sum,0)

        ## final loss 
        if params['use_targ_term']:
            loss = vanilla_loss + params['ratio'] * targeted_regularization+params['b_ratio']*tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)
        else:
            loss = vanilla_loss + params['b_ratio']*tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)
        
        ## add final loss 
        model.add_loss(loss)
        
        ##
        self.model = model
        self.params = params
        
    
    def support_ite(self):
        return True 
    
    def predict_ite(self, X):
        dummy = np.zeros((X.shape[0],))
        preds = self.model.predict([X,dummy,dummy])
        if self.params['scale_preds']:
            preds[:, 0] = np.squeeze(self.y_scaler.inverse_transform(preds[:, 0].reshape(-1,1).copy()),axis=-1)
            preds[:, 1] = np.squeeze(self.y_scaler.inverse_transform(preds[:, 1].reshape(-1,1).copy()),axis=-1)
        return (preds[:, 1] - preds[:, 0])
    
    def predict_ate(self, X):
        return np.mean(self.predict_ite(X))
    
    def fit(self, X, treatment, y):
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        treatment = treatment.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        if self.params['scale_preds']:
            self.y_scaler = StandardScaler().fit(y)
            y = self.y_scaler.transform(y) 
        
        start_time = time.time()
        if self.params['optim'] == 'adam':
            self.model.compile(
            optimizer=Adam(lr=self.params['learning_rate']))
            adam_callbacks = [
                TerminateOnNaN(),
                EarlyStopping(monitor='val_loss', patience=40, min_delta=0., restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=self.params['verbose'], mode='auto',
                                  min_delta=1e-8, cooldown=0, min_lr=0)
        
            ]
            dummy = np.zeros((X.shape[0],))
            self.history = self.model.fit([X, y, treatment], dummy, callbacks=adam_callbacks,
                          validation_split=self.params['val_split'],
                          epochs=self.params['epochs'],
                          batch_size=int(X.shape[0]*self.params['bs_ratio']), 
                          verbose=self.params['verbose'])
            
        elif self.params['optim'] == 'sgd':
            sgd_callbacks = [
                TerminateOnNaN(),
                EarlyStopping(monitor='val_loss', patience=40, min_delta=0. , restore_best_weights=True),
                #ModelCheckpoint('bcauss.h5', save_best_only=True, save_weights_only=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=self.params['verbose'], mode='auto',
                                  min_delta=0., cooldown=0, min_lr=0)
            ]
        
            self.model.compile(optimizer=SGD(lr=self.params['learning_rate'], momentum=self.params['momentum'], nesterov=True))
            dummy = np.zeros((X.shape[0],))
            self.history = self.model.fit([X, y, treatment], dummy, callbacks=sgd_callbacks,
                          validation_split=self.params['val_split'],
                          epochs=self.params['epochs'], #300
                          batch_size=int(X.shape[0]*self.params['bs_ratio']), 
                          verbose=self.params['verbose'])  
            
        else:
            raise Exception("optim <"+str(self.params['optim'])+"> not supported!")
        
        #
        elapsed_time = time.time() - start_time
        if self.params['verbose']:
            print("***************************** elapsed_time is: ", elapsed_time)
        #
        
        
        
        
        
        
        
        
        
