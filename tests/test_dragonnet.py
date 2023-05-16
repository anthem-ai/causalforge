import pandas as pd 
import numpy as np 
from causalforge.model import Model , PROBLEM_TYPE
from causalforge.data_loader import DataLoader 
from causalforge.metrics import eps_ATE_diff, PEHE_with_ite

def test_train():
     
    # load IHDP dataset 
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r
    
    # model 
    params={}
    params['input_dim'] = X_tr.shape[1] 
    params['neurons_per_layer'] = 200 
    params['reg_l2'] = 0.01
    params['targeted_reg'] = True 
    params['verbose'] = True 
    params['val_split'] = 0.2
    params['ratio'] = 1.0
    params['batch_size'] = 64
    params['epochs'] = 100
    params['learning_rate'] = 1e-5
    params['momentum'] = 0.9
    params['use_adam'] = True
    params['adam_epochs'] = 30
    params['adam_learning_rate'] = 1e-3
    
    dragonnet = Model.create_model("dragonnet", 
                                   params,
                                   problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                                   multiple_treatments=False)
    
    assert dragonnet.model is not None 
    
    
    eps_ATE_tr, eps_ATE_te = [], []
    eps_PEHE_tr, eps_PEHE_te = [] , []

    # train
    idx = 1 
    t_tr, y_tr, x_tr, mu0tr, mu1tr = T_tr[:,idx] , YF_tr[:,idx], X_tr[:,:,idx], mu_0_tr[:,idx], mu_1_tr[:,idx]
    t_te, y_te, x_te, mu0te, mu1te = T_te[:,idx] , YF_te[:,idx], X_te[:,:,idx], mu_0_te[:,idx], mu_1_te[:,idx]

    dragonnet.fit(x_tr,t_tr,y_tr)
    
    # test 
    pred_test_ite =  dragonnet.predict_ite(x_te)
    
    # error 
    ATE_truth_tr = (mu1tr - mu0tr).mean()
    ATE_truth_te = (mu1te - mu0te).mean()

    ITE_truth_tr = (mu1tr - mu0tr)
    ITE_truth_te = (mu1te - mu0te)

    eps_ATE_tr.append( eps_ATE_diff( dragonnet.predict_ate(x_tr,t_tr,y_tr), ITE_truth_tr) )
    eps_ATE_te.append( eps_ATE_diff( dragonnet.predict_ate(x_te,t_te,y_te), ITE_truth_te) )

    eps_PEHE_tr.append( PEHE_with_ite( dragonnet.predict_ite(x_tr), ITE_truth_tr, sqrt=True))
    eps_PEHE_te.append( PEHE_with_ite( dragonnet.predict_ite(x_te), ITE_truth_te , sqrt=True))
    
    print("*******************************************************************")
    print(pd.DataFrame([[np.mean(eps_ATE_tr),np.mean(eps_ATE_te),np.mean(eps_PEHE_tr),np.mean(eps_PEHE_te)]],
             columns=['eps_ATE_tr','eps_ATE_te','eps_PEHE_tr','eps_PEHE_te'],
             index=['DragonNet']))
    print("*******************************************************************")
    
    
    
    
    
    
