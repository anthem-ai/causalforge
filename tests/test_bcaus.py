import pandas as pd 
import numpy as np 
from sklearn import metrics
from causalforge.model import Model , PROBLEM_TYPE
from causalforge.data_loader import DataLoader 
from causalforge.metrics import eps_ATE_diff, PEHE_with_ite



def test_bcaus_dr():
     
    # load IHDP dataset 
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r
    
    # model 
    params={}
    params['input_dim'] = X_tr.shape[1] 
    
    bcaus_dr = Model.create_model("bcaus_dr", 
                                   params,
                                   problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                                   multiple_treatments=False)
    
    eps_ATE_tr, eps_ATE_te = [], []
    eps_PEHE_tr, eps_PEHE_te = [] , []

    # train
    idx = 1 
    t_tr, y_tr, x_tr, mu0tr, mu1tr = T_tr[:,idx] , YF_tr[:,idx], X_tr[:,:,idx], mu_0_tr[:,idx], mu_1_tr[:,idx]
    t_te, y_te, x_te, mu0te, mu1te = T_te[:,idx] , YF_te[:,idx], X_te[:,:,idx], mu_0_te[:,idx], mu_1_te[:,idx]

    bcaus_dr.fit(x_tr,t_tr,y_tr)
    
    # error 
    ATE_truth_tr = (mu1tr - mu0tr).mean()
    ATE_truth_te = (mu1te - mu0te).mean()

    eps_ATE_tr.append( eps_ATE_diff( bcaus_dr.predict_ate(x_tr,t_tr,y_tr), ATE_truth_tr) )
    eps_ATE_te.append( eps_ATE_diff( bcaus_dr.predict_ate(x_te,t_te,y_te), ATE_truth_te) )
    
    print("*******************************************************************")
    print(pd.DataFrame([[np.mean(eps_ATE_tr),np.mean(eps_ATE_te)]],
             columns=['eps_ATE_tr','eps_ATE_te'],
             index=['BCAUSS']))
    print("*******************************************************************")

def test_bcaus():
    
    
    def compute_auc(x_te,t_te):
        pred_test_propensity =  bcaus.predict_proba(x_te)
        
        for i in range(pred_test_propensity.shape[0]):
            assert pred_test_propensity[i][0]+pred_test_propensity[i][1] == 1.0 
            
        pred_1 = np.apply_along_axis(lambda x: x[1], 1, pred_test_propensity)
        fpr, tpr, thresholds = metrics.roc_curve(t_te, pred_1)
        AUC_te = metrics.auc(fpr, tpr)
        return AUC_te
     
    # load IHDP dataset 
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r
    
    # model 
    params={}
    bcaus = Model.create_model("bcaus", 
                                   params,
                                   problem_type=PROBLEM_TYPE.PROPENSITY_ESTIMATION, 
                                   multiple_treatments=False)
    
    eps_ATE_tr, eps_ATE_te = [], []
    eps_PEHE_tr, eps_PEHE_te = [] , []

    # train
    idx = 1 
    t_tr, y_tr, x_tr, mu0tr, mu1tr = T_tr[:,idx] , YF_tr[:,idx], X_tr[:,:,idx], mu_0_tr[:,idx], mu_1_tr[:,idx]
    t_te, y_te, x_te, mu0te, mu1te = T_te[:,idx] , YF_te[:,idx], X_te[:,:,idx], mu_0_te[:,idx], mu_1_te[:,idx]

    bcaus.fit(x_tr,t_tr)
    
    # test 
    AUC_tr = compute_auc(x_tr,t_tr)
    AUC_te = compute_auc(x_te,t_te)
    
    print("*******************************************************************")
    print(pd.DataFrame([[AUC_tr,AUC_te]],
             columns=['AUC_tr','AUC_te'],
             index=['BCAUS']))
    print("*******************************************************************")
     
    
    

