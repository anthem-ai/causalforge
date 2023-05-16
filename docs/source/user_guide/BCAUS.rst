BCAUS
=====

**Reference**: `Belthangady et al, Minimizing bias in massive multi-arm
observational studies with BCAUS: balancing covariates automatically
using supervision, BMC Medical Research Methodology
2021 <https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-021-01383-x>`__

BCAUS on IHDP
-----

.. code:: python

    from causalforge.model import Model , PROBLEM_TYPE
    from causalforge.data_loader import DataLoader 
    
    # load IHDP dataset 
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r
    
    # model 
    params={}
        
        
    bcaus_dr = Model.create_model("bcaus_dr",
                                   params,
                                   problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                                   multiple_treatments=False)


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
        bcaus_dr.fit(x_tr,t_tr,y_tr)
    
        # Validate your method test-set ... 
        ATE_truth_tr = (mu1tr - mu0tr).mean()
        ATE_truth_te = (mu1te - mu0te).mean()
        
        ITE_truth_tr = (mu1tr - mu0tr)
        ITE_truth_te = (mu1te - mu0te)
        
        eps_ATE_tr.append( eps_ATE_diff( bcaus_dr.predict_ate(x_tr,t_tr,y_tr), ATE_truth_tr) )
        eps_ATE_te.append( eps_ATE_diff( bcaus_dr.predict_ate(x_te,t_te,y_te), ATE_truth_te) )
            

Results
-------

.. code:: python

    import pandas as pd 
    
    pd.DataFrame([[np.mean(eps_ATE_tr),np.mean(eps_ATE_te)]],
                 columns=['eps_ATE_tr','eps_ATE_te'], 
                 index=['BCAUS_DR'])




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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>BCAUS_DR</th>
          <td>0.012877</td>
          <td>0.260656</td>
        </tr>
      </tbody>
    </table>
    </div>



