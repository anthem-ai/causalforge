BCAUSS
======

**Reference**: `Gino Tesei et al, Learning end-to-end patient
representations through self-supervised covariate balancing for causal
treatment effect estimation, Journal of Biomedical Informatics
2023 <https://www.sciencedirect.com/science/article/pii/S1532046423000606/pdfft?md5=923768a5e1b27765e9da9ac13c0477aa&pid=1-s2.0-S1532046423000606-main.pdf>`__

BCAUSS on IHDP
--------------

.. code:: python

    from causalforge.model import Model , PROBLEM_TYPE
    from causalforge.data_loader import DataLoader 
    
    # load IHDP dataset 
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r
    
    # model 
    params={}
    params['input_dim'] = X_tr.shape[1] 
        
        
    bcauss = Model.create_model("bcauss",
                                   params,
                                   problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                                   multiple_treatments=False)
    
    bcauss.model.summary()


.. parsed-literal::

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input (InputLayer)             [(None, 25)]         0           []                               
                                                                                                      
     dense (Dense)                  (None, 200)          5200        ['input[0][0]']                  
                                                                                                      
     dense_1 (Dense)                (None, 200)          40200       ['dense[0][0]']                  
                                                                                                      
     dense_2 (Dense)                (None, 200)          40200       ['dense_1[0][0]']                
                                                                                                      
     dense_4 (Dense)                (None, 100)          20100       ['dense_2[0][0]']                
                                                                                                      
     dense_5 (Dense)                (None, 100)          20100       ['dense_2[0][0]']                
                                                                                                      
     dense_6 (Dense)                (None, 100)          10100       ['dense_4[0][0]']                
                                                                                                      
     dense_7 (Dense)                (None, 100)          10100       ['dense_5[0][0]']                
                                                                                                      
     dense_3 (Dense)                (None, 1)            201         ['dense_2[0][0]']                
                                                                                                      
     y0_predictions (Dense)         (None, 1)            101         ['dense_6[0][0]']                
                                                                                                      
     y1_predictions (Dense)         (None, 1)            101         ['dense_7[0][0]']                
                                                                                                      
     epsilon_layer (EpsilonLayer)   (None, 1)            1           ['dense_3[0][0]']                
                                                                                                      
     y_true (InputLayer)            [(None, 1)]          0           []                               
                                                                                                      
     t_true (InputLayer)            [(None, 1)]          0           []                               
                                                                                                      
     concatenate (Concatenate)      (None, 4)            0           ['y0_predictions[0][0]',         
                                                                      'y1_predictions[0][0]',         
                                                                      'dense_3[0][0]',                
                                                                      'epsilon_layer[0][0]']          
                                                                                                      
     tf.__operators__.add (TFOpLamb  (None, 1)           0           ['dense_3[0][0]']                
     da)                                                                                              
                                                                                                      
     tf.math.truediv (TFOpLambda)   (None, 1)            0           ['tf.__operators__.add[0][0]']   
                                                                                                      
     tf.math.subtract_7 (TFOpLambda  (None, 1)           0           ['t_true[0][0]']                 
     )                                                                                                
                                                                                                      
     tf.math.subtract_8 (TFOpLambda  (None, 1)           0           ['tf.math.truediv[0][0]']        
     )                                                                                                
                                                                                                      
     tf.math.truediv_3 (TFOpLambda)  (None, 1)           0           ['t_true[0][0]',                 
                                                                      'tf.math.truediv[0][0]']        
                                                                                                      
     tf.math.truediv_4 (TFOpLambda)  (None, 1)           0           ['tf.math.subtract_7[0][0]',     
                                                                      'tf.math.subtract_8[0][0]']     
                                                                                                      
     tf.split (TFOpLambda)          [(None, 1)]          0           ['tf.math.truediv_3[0][0]']      
                                                                                                      
     tf.split_1 (TFOpLambda)        [(None, 1)]          0           ['tf.math.truediv_4[0][0]']      
                                                                                                      
     tf.concat (TFOpLambda)         (None, 25)           0           ['tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]',               
                                                                      'tf.split[0][0]']               
                                                                                                      
     tf.concat_1 (TFOpLambda)       (None, 25)           0           ['tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]',             
                                                                      'tf.split_1[0][0]']             
                                                                                                      
     tf.math.subtract_9 (TFOpLambda  (None, 1)           0           ['t_true[0][0]']                 
     )                                                                                                
                                                                                                      
     tf.math.subtract_10 (TFOpLambd  (None, 1)           0           ['tf.math.truediv[0][0]']        
     a)                                                                                               
                                                                                                      
     tf.math.multiply_4 (TFOpLambda  (None, 25)          0           ['tf.concat[0][0]',              
     )                                                                'input[0][0]']                  
                                                                                                      
     tf.math.truediv_5 (TFOpLambda)  (None, 1)           0           ['t_true[0][0]',                 
                                                                      'tf.math.truediv[0][0]']        
                                                                                                      
     tf.math.multiply_5 (TFOpLambda  (None, 25)          0           ['tf.concat_1[0][0]',            
     )                                                                'input[0][0]']                  
                                                                                                      
     tf.math.truediv_7 (TFOpLambda)  (None, 1)           0           ['tf.math.subtract_9[0][0]',     
                                                                      'tf.math.subtract_10[0][0]']    
                                                                                                      
     tf.math.reduce_sum_3 (TFOpLamb  (25,)               0           ['tf.math.multiply_4[0][0]']     
     da)                                                                                              
                                                                                                      
     tf.math.reduce_sum_4 (TFOpLamb  (1,)                0           ['tf.math.truediv_5[0][0]']      
     da)                                                                                              
                                                                                                      
     tf.math.reduce_sum_5 (TFOpLamb  (25,)               0           ['tf.math.multiply_5[0][0]']     
     da)                                                                                              
                                                                                                      
     tf.math.reduce_sum_6 (TFOpLamb  (1,)                0           ['tf.math.truediv_7[0][0]']      
     da)                                                                                              
                                                                                                      
     tf.math.subtract_1 (TFOpLambda  (None, 1)           0           ['y_true[0][0]',                 
     )                                                                'y0_predictions[0][0]']         
                                                                                                      
     tf.math.subtract_2 (TFOpLambda  (None, 1)           0           ['y_true[0][0]',                 
     )                                                                'y1_predictions[0][0]']         
                                                                                                      
     tf.math.truediv_6 (TFOpLambda)  (25,)               0           ['tf.math.reduce_sum_3[0][0]',   
                                                                      'tf.math.reduce_sum_4[0][0]']   
                                                                                                      
     tf.math.truediv_8 (TFOpLambda)  (25,)               0           ['tf.math.reduce_sum_5[0][0]',   
                                                                      'tf.math.reduce_sum_6[0][0]']   
                                                                                                      
     tf.math.subtract (TFOpLambda)  (None, 1)            0           ['t_true[0][0]']                 
                                                                                                      
     tf.math.square (TFOpLambda)    (None, 1)            0           ['tf.math.subtract_1[0][0]']     
                                                                                                      
     tf.math.square_1 (TFOpLambda)  (None, 1)            0           ['tf.math.subtract_2[0][0]']     
                                                                                                      
     tf.convert_to_tensor_2 (TFOpLa  (25,)               0           ['tf.math.truediv_6[0][0]']      
     mbda)                                                                                            
                                                                                                      
     tf.cast (TFOpLambda)           (25,)                0           ['tf.math.truediv_8[0][0]']      
                                                                                                      
     tf.math.multiply (TFOpLambda)  (None, 1)            0           ['tf.math.subtract[0][0]',       
                                                                      'tf.math.square[0][0]']         
                                                                                                      
     tf.math.multiply_1 (TFOpLambda  (None, 1)           0           ['t_true[0][0]',                 
     )                                                                'tf.math.square_1[0][0]']       
                                                                                                      
     tf.math.squared_difference (TF  (25,)               0           ['tf.convert_to_tensor_2[0][0]', 
     OpLambda)                                                        'tf.cast[0][0]']                
                                                                                                      
     tf.math.reduce_sum_1 (TFOpLamb  ()                  0           ['tf.math.multiply[0][0]']       
     da)                                                                                              
                                                                                                      
     tf.math.reduce_sum_2 (TFOpLamb  ()                  0           ['tf.math.multiply_1[0][0]']     
     da)                                                                                              
                                                                                                      
     tf.math.reduce_mean (TFOpLambd  ()                  0           ['tf.math.squared_difference[0][0
     a)                                                              ]']                              
                                                                                                      
     tf.__operators__.add_1 (TFOpLa  ()                  0           ['tf.math.reduce_sum_1[0][0]',   
     mbda)                                                            'tf.math.reduce_sum_2[0][0]']   
                                                                                                      
     tf.math.multiply_6 (TFOpLambda  ()                  0           ['tf.math.reduce_mean[0][0]']    
     )                                                                                                
                                                                                                      
     tf.__operators__.add_3 (TFOpLa  ()                  0           ['tf.__operators__.add_1[0][0]', 
     mbda)                                                            'tf.math.multiply_6[0][0]']     
                                                                                                      
     add_loss (AddLoss)             ()                   0           ['tf.__operators__.add_3[0][0]'] 
                                                                                                      
    ==================================================================================================
    Total params: 146,404
    Trainable params: 146,404
    Non-trainable params: 0
    __________________________________________________________________________________________________


Parameters
----------

-  **input_dim**: number of inputs
-  **neurons_per_layer**: number of neurons per layer (by default,
   ``200``)
-  **act_fn**: activation function [``relu``\ \|\ ``elu``\ \|\ ``tanh``]
   (by default, ``relu``)
-  **reg_l2**: L2 regularization (by default, ``0.01``)
-  **verbose**: verbose (by default, ``True``)
-  **val_split**: validation split ratio (by default, ``0.22``)
-  **ratio**: relative importance of the targeted regularization term,
   if adopted (by default, ``1.0``)
-  **optim**: optimizer [``sgd``\ \|\ ``adam``] (by default, ``sgd``)
-  **epochs**: number of epochs (by default, ``500``)
-  **learning_rate**: learning rate (by default, ``1e-5``)
-  **momentum**: momentum (by default, ``0.9``)
-  **use_bce**: to use the binary cross-entropy to train the propensity
   score estimator (by default, ``False``)
-  **norm_bal_term**: to normalize the auto-balancing regularization
   term (by default, ``True``)
-  **use_targ_term**: to use the targeted regularization objective (by
   default, ``False``)
-  **b_ratio**: relative importance of the auto-balancing regularization
   objective (by default, ``1.0``)
-  **bs_ratio**: percentage of train-set to use as batch-size (by
   default, ``1.0``)
-  **scale_preds**: to scale predictions (by default, ``True``)

Training
--------

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
        bcauss.fit(x_tr,t_tr,y_tr)
    
        # Validate your method test-set ... 
        ATE_truth_tr = (mu1tr - mu0tr).mean()
        ATE_truth_te = (mu1te - mu0te).mean()
        
        ITE_truth_tr = (mu1tr - mu0tr)
        ITE_truth_te = (mu1te - mu0te)
        
        eps_ATE_tr.append( eps_ATE_diff( bcauss.predict_ate(x_tr,t_tr,y_tr), ATE_truth_tr) )
        eps_ATE_te.append( eps_ATE_diff( bcauss.predict_ate(x_te,t_te,y_te), ATE_truth_te) )
        
        eps_PEHE_tr.append( PEHE_with_ite( bcauss.predict_ite(x_tr), ITE_truth_tr, sqrt=True))
        eps_PEHE_te.append( PEHE_with_ite( bcauss.predict_ite(x_te), ITE_truth_te , sqrt=True))
            


.. parsed-literal::

    Epoch 1/500

    ...

    Epoch 499/500
    1/1 [==============================] - 0s 43ms/step - loss: 14.0257 - val_loss: 16.1863 - lr: 1.0000e-05
    Epoch 500/500
    1/1 [==============================] - 0s 41ms/step - loss: 14.0177 - val_loss: 16.1839 - lr: 1.0000e-05
    ***************************** elapsed_time is:  21.736793041229248
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
                 index=['BCAUSS'])




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
          <th>BCAUSS</th>
          <td>0.067121</td>
          <td>0.084479</td>
          <td>0.567214</td>
          <td>0.510539</td>
        </tr>
      </tbody>
    </table>
    </div>



ITE distribution: learned vs. ground truth
------------------------------------------

Learned
~~~~~~~

.. code:: python

    from causalforge.utils import plot_ite_distribution
    
    plot_ite_distribution(bcauss.predict_ite(x_te))




.. image:: bcauss_output_9_2.png


Ground Truth
~~~~~~~~~~~~

.. code:: python

    from causalforge.utils import plot_ite_distribution
    
    plot_ite_distribution(ITE_truth_te)




.. image:: bcauss_output_11_1.png


