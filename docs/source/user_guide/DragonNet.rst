DragonNet
=========

**Reference**: `Claudia Shi et al, Adapting Neural Networks for the
Estimation of Treatment Effects, NeurIPS
2019 <https://arxiv.org/pdf/1906.02120v2.pdf>`__

**Implementation remarks**: our implementation is exactly the same of
the original paper with the exception of a
*sklearn.preprocessing.StandardScaler* which was originally used to
scale predictions.

DragonNet on IHDP
-----------------

.. code:: python

    from causalforge.model import Model , PROBLEM_TYPE
    from causalforge.data_loader import DataLoader 
    
    # load IHDP dataset 
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r
    
    # model 
    params={}
    params['input_dim'] = X_tr.shape[1] 
        
        
    dragonnet = Model.create_model("dragonnet",
                                   params,
                                   problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                                   multiple_treatments=False)
    
    dragonnet.model.summary()


.. parsed-literal::

    2023-05-05 16:08:33.088174: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input (InputLayer)             [(None, 25)]         0           []                               
                                                                                                      
     dense (Dense)                  (None, 200)          5200        ['input[0][0]']                  
                                                                                                      
     dense_1 (Dense)                (None, 200)          40200       ['dense[0][0]']                  
                                                                                                      
     dense_2 (Dense)                (None, 200)          40200       ['dense_1[0][0]']                
                                                                                                      


.. parsed-literal::

    2023-05-05 16:08:36.229178: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

     dense_4 (Dense)                (None, 100)          20100       ['dense_2[0][0]']                
                                                                                                      
     dense_5 (Dense)                (None, 100)          20100       ['dense_2[0][0]']                
                                                                                                      
     dense_6 (Dense)                (None, 100)          10100       ['dense_4[0][0]']                
                                                                                                      
     dense_7 (Dense)                (None, 100)          10100       ['dense_5[0][0]']                
                                                                                                      
     dense_3 (Dense)                (None, 1)            201         ['dense_2[0][0]']                
                                                                                                      
     y0_predictions (Dense)         (None, 1)            101         ['dense_6[0][0]']                
                                                                                                      
     y1_predictions (Dense)         (None, 1)            101         ['dense_7[0][0]']                
                                                                                                      
     epsilon_layer (EpsilonLayer)   (None, 1)            1           ['dense_3[0][0]']                
                                                                                                      
     concatenate (Concatenate)      (None, 4)            0           ['y0_predictions[0][0]',         
                                                                      'y1_predictions[0][0]',         
                                                                      'dense_3[0][0]',                
                                                                      'epsilon_layer[0][0]']          
                                                                                                      
    ==================================================================================================
    Total params: 146,404
    Trainable params: 146,404
    Non-trainable params: 0
    __________________________________________________________________________________________________


.. code:: python

    from causalforge.metrics import eps_ATE_diff, PEHE_with_ite
    import numpy as np
    
    experiment_ids = [1,10,400]
    
    eps_ATE_tr, eps_ATE_te = [], []
    eps_PEHE_tr, eps_PEHE_te = [] , [] 
    
    
    
    for idx in experiment_ids:    
        t_tr, y_tr, x_tr, mu0tr, mu1tr = T_tr[:,idx] , YF_tr[:,idx], X_tr[:,:,idx], mu_0_tr[:,idx], mu_1_tr[:,idx] 
        t_te, y_te, x_te, mu0te, mu1te = T_te[:,idx] , YF_te[:,idx], X_te[:,:,idx], mu_0_te[:,idx], mu_1_te[:,idx]  
        
        
        # Train your causal method on train-set ...
        dragonnet.fit(x_tr,t_tr,y_tr)
    
        # Validate your method test-set ... 
        ATE_truth_tr = (mu1tr - mu0tr).mean()
        ATE_truth_te = (mu1te - mu0te).mean()
        
        ITE_truth_tr = (mu1tr - mu0tr)
        ITE_truth_te = (mu1te - mu0te)
        
        eps_ATE_tr.append( eps_ATE_diff( dragonnet.predict_ite(x_tr), ITE_truth_tr) )
        eps_ATE_te.append( eps_ATE_diff( dragonnet.predict_ite(x_te), ITE_truth_te) )
        
        eps_PEHE_tr.append( PEHE_with_ite( dragonnet.predict_ite(x_tr), ITE_truth_tr, sqrt=True))
        eps_PEHE_te.append( PEHE_with_ite(dragonnet.predict_ite(x_te), ITE_truth_te , sqrt=True))
            


.. parsed-literal::

    Epoch 1/30


.. parsed-literal::

    /Users/AG62216/opt/anaconda3/envs/causalforge/lib/python3.10/site-packages/keras/optimizers/optimizer_v2/adam.py:114: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super().__init__(name, **kwargs)


.. parsed-literal::

    9/9 [==============================] - 2s 43ms/step - loss: 1837.5880 - regression_loss: 847.2359 - binary_classification_loss: 35.7781 - treatment_accuracy: 0.7672 - track_epsilon: 0.0607 - val_loss: 402.8440 - val_regression_loss: 140.5852 - val_binary_classification_loss: 21.5544 - val_treatment_accuracy: 0.8006 - val_track_epsilon: 0.0610 - lr: 0.0010
    Epoch 2/30
    9/9 [==============================] - 0s 7ms/step - loss: 355.1468 - regression_loss: 146.8956 - binary_classification_loss: 30.7850 - treatment_accuracy: 0.8042 - track_epsilon: 0.0616 - val_loss: 186.1139 - val_regression_loss: 57.0438 - val_binary_classification_loss: 20.9781 - val_treatment_accuracy: 0.8006 - val_track_epsilon: 0.0606 - lr: 0.0010
    Epoch 3/30
    9/9 [==============================] - 0s 6ms/step - loss: 254.1957 - regression_loss: 107.0528 - binary_classification_loss: 30.1929 - treatment_accuracy: 0.8069 - track_epsilon: 0.0597 - val_loss: 217.4558 - val_regression_loss: 70.0112 - val_binary_classification_loss: 21.2913 - val_treatment_accuracy: 0.8006 - val_track_epsilon: 0.0595 - lr: 0.0010
    Epoch 4/30
    9/9 [==============================] - 0s 7ms/step - loss: 226.9801 - regression_loss: 90.6360 - binary_classification_loss: 29.8652 - treatment_accuracy: 0.8069 - track_epsilon: 0.0597 - val_loss: 205.2203 - val_regression_loss: 63.8470 - val_binary_classification_loss: 20.7406 - val_treatment_accuracy: 0.8006 - val_track_epsilon: 0.0594 - lr: 0.0010
    Epoch 1/100


.. parsed-literal::

    /Users/AG62216/opt/anaconda3/envs/causalforge/lib/python3.10/site-packages/keras/optimizers/optimizer_v2/gradient_descent.py:111: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super().__init__(name, **kwargs)


.. parsed-literal::

    9/9 [==============================] - 2s 46ms/step - loss: 211.6185 - regression_loss: 84.1101 - binary_classification_loss: 29.5777 - treatment_accuracy: 0.8096 - track_epsilon: 0.0494 - val_loss: 187.8485 - val_regression_loss: 57.8545 - val_binary_classification_loss: 20.9967 - val_treatment_accuracy: 0.8006 - val_track_epsilon: 0.0354 - lr: 1.0000e-05
    Epoch 2/100
    9/9 [==============================] - 0s 8ms/step - loss: 201.3698 - regression_loss: 79.9347 - binary_classification_loss: 29.4149 - treatment_accuracy: 0.8123 - track_epsilon: 0.0361 - val_loss: 173.2833 - val_regression_loss: 52.0208 - val_binary_classification_loss: 20.9611 - val_treatment_accuracy: 0.8006 - val_track_epsilon: 0.0361 - lr: 1.0000e-05
    Epoch 3/100
    ...
    9/9 [==============================] - 0s 6ms/step - loss: 148.3127 - regression_loss: 55.6120 - binary_classification_loss: 27.1335 - treatment_accuracy: 0.8009 - track_epsilon: 0.0145 - val_loss: 218.1645 - val_regression_loss: 106.4683 - val_binary_classification_loss: 18.0388 - val_treatment_accuracy: 0.9062 - val_track_epsilon: 0.0144 - lr: 6.2500e-07
    21/21 [==============================] - 0s 1ms/step
    3/3 [==============================] - 0s 2ms/step
    21/21 [==============================] - 0s 1ms/step
    3/3 [==============================] - 0s 1ms/step


Results
-------

.. code:: python

    import pandas as pd 
    
    pd.DataFrame([[np.mean(eps_ATE_tr),np.mean(eps_ATE_te),np.mean(eps_PEHE_tr),np.mean(eps_PEHE_te)]],
                 columns=['eps_ATE_tr','eps_ATE_te','eps_PEHE_tr','eps_PEHE_te'], 
                 index=['DragonNet'])




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>eps_ATE_tr</th>
          <th>eps_ATE_te</th>
          <th>eps_PEHE_tr</th>
          <th>eps_PEHE_te</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>DragonNet</th>
          <td>0.091358</td>
          <td>0.080693</td>
          <td>0.654876</td>
          <td>0.649264</td>
        </tr>
      </tbody>
    </table>
    </div>


