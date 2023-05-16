"""
Code Skeleton from: https://github.com/gstef80/bcaus_nma/blob/main/bcaus.py
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from causalforge.model import Propensity_Estimator , Model , PROBLEM_TYPE


class BCAUS_DR(Model):
    
    def build(self,params):
        self.bcaus = Model.create_model("bcaus", 
                                       params,
                                       problem_type=PROBLEM_TYPE.PROPENSITY_ESTIMATION, 
                                       multiple_treatments=False)
        self.params = params
    
    
    def fit(self, X, treatment, y):
        """
        Fits the model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        self.bcaus.fit(X,treatment)
    
        treated_idx=np.where(treatment==1)[0]
        control_idx=np.where(treatment==0)[0]
        
        # Fit estimators for DR
        params={"alpha":[0.001,0.01,0.1]}
        self.estimator_t = GridSearchCV(Ridge(), param_grid=params, cv=3, n_jobs=3)
        self.estimator_c = GridSearchCV(Ridge(), param_grid=params, cv=3, n_jobs=3)
        self.estimator_t.fit(X[treated_idx,:], y[treated_idx])
        self.estimator_c.fit(X[control_idx,:], y[control_idx]) 
    
   
    def support_ite(self):
        """
        Whether the model supports individual treatment effect ("ITE") estimation 

        Returns:
            (Boolean): Whether the model supports ITE estimation
        """
        return False  
    
    
    def predict_ite(self, X):
        """
        Predicts the individual treatment effect ("ITE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.array): treatment effect vector
        """
        raise Exception('ITE not supported') 
    
    
    def predict_ate(self, X,treatment,y):
        """
        Predicts the average treatment effect ("ATE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.array): treatment effect vector
        """
        scores = self.bcaus.predict_proba(X)[:,1]
        weights = (treatment / scores + (1 - treatment) / (1 - scores))
        
        treated_idx=np.where(treatment==1)[0]
        control_idx=np.where(treatment==0)[0]
        
        treatment_yf_pred= self.estimator_t.predict(X[treated_idx,:])
        treatment_ycf_pred = self.estimator_c.predict(X[treated_idx,:])
        control_yf_pred = self.estimator_c.predict(X[control_idx,:])
        control_ycf_pred = self.estimator_t.predict(X[control_idx,:])
    
        treatment_ite = (y[treated_idx]/scores[treated_idx]
                            -treatment_yf_pred*(1-scores[treated_idx])/scores[treated_idx]
                            -treatment_ycf_pred)
        control_ite = control_ycf_pred-(y[control_idx]/(1-scores[control_idx])
                                        -control_yf_pred*scores[control_idx]/(1-scores[control_idx]))
        
        
        est_ate = np.mean(np.array(list(treatment_ite)+list(control_ite)))
        
        return est_ate  


class Propensity(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        """
        Define Pytorch model
        :param input_size: Number of covariates
        :param hidden_size: Number of hidden layer neurons
        :param dropout_prob: Dropout probability
        """
        super(Propensity, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in')
        self.do1 = nn.Dropout(p=dropout_prob)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in')
        self.do2 = nn.Dropout(p=dropout_prob)
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.do1(self.layer1(x)))
        x = F.relu(self.do2(self.layer2(x)))
        x = torch.sigmoid(self.layer3(x))
        return x


class BCAUS(Propensity_Estimator,BaseEstimator, ClassifierMixin):
    
    def build(self,user_params):
        
        params = {
            'random_state': 271, 
            "hidden_layer_size": None, 
            "batch_size": None, 
            "shuffle": True,
            "learning_rate_init": 0.001,
            "nu": 1, 
            "max_iter": 100,
            "alpha": 0.0,
            "dropout": 0,
            "eps": 1e-5,
            "early_stopping": False, 
            "n_iter_no_change": 10, 
            "balance_threshold": 0.1, 
            "device": 'cpu', 
            "verbose": False, 
            "logger": None
        }
        
        for k in params:
            params[k] = user_params.get(k,params[k])
            setattr(self, k, params[k])
            
        self.params = params 
        
        
    def score(self, X, y):
        """
        Returns number of balanced covariates instead of accuracy score since
        during cross-validation, this is the metric we want to optimize.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        X = torch.tensor(X, dtype=torch.float32, device=torch.device(self.device))
        y = torch.tensor(y, dtype=torch.float32, device=torch.device(self.device))
        num_balanced = self._balanced_cov(X, y)

        return num_balanced
    
    def fit(self, X, treatment):
        """
        Fits the model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
        """
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if self.verbose and (self.logger is None):
            raise ValueError('If verbose is set to True, logger should be specified')

        device = torch.device(self.device)

        # Check that X and y have correct shape
        X, y = check_X_y(X, treatment)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)

        num_features = X.shape[1]

        if self.hidden_layer_size is not None:
            self.model = Propensity(num_features, self.hidden_layer_size, self.dropout)
        else:
            self.model = Propensity(num_features, 2 * num_features, self.dropout)

        self.model.to(device)

        criterion = (nn.BCELoss(), nn.MSELoss())
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, betas=(0.5, 0.999),
                               weight_decay=self.alpha)

        if self.batch_size is not None:
            ds = TensorDataset(X, y)
            trainloader = DataLoader(ds, self.batch_size, shuffle=self.shuffle, drop_last=True)

        prop_loss = []
        cov_loss = []
        num_balanced = []

        for i in range(self.max_iter):

            if self.batch_size is not None:
                prop_epoch_loss = 0.0
                cov_epoch_loss = 0.0
                for X0, y0 in trainloader:
                    loss_prop, loss_cov = self._step(X0, y0, optimizer, criterion)
                    prop_epoch_loss += loss_prop
                    cov_epoch_loss += loss_cov

                prop_loss.append(prop_epoch_loss / len(trainloader))
                cov_loss.append(cov_epoch_loss / len(trainloader))
            else:
                loss_prop, loss_cov = self._step(X, y, optimizer, criterion)
                prop_loss.append(loss_prop)
                cov_loss.append(loss_cov)

            num_balanced.append(self._balanced_cov(X, y))

            if self.early_stopping and len(num_balanced) > self.n_iter_no_change + 1:
                if ((num_balanced[-self.n_iter_no_change:] == num_balanced[-self.n_iter_no_change - 1:-1])
                        and (num_balanced[-1] == num_features)):
                    if self.verbose:
                        self.logger.info('All covariates balanced at epoch {}'.format(i))
                    break

            if self.verbose:
                if i % 50 == 0:
                    self.logger.info('Epoch ={}: Propensity Loss ={}, Covariate Loss ={}, Balanced covs ={}'
                                     .format(i, prop_loss[-1], cov_loss[-1], num_balanced[-1]))

        self.model = self.model.eval()
        self.loss_stats_ = (prop_loss, cov_loss, num_balanced)

        if self.verbose:
            self.logger.info('Number of balanced covariates at end of training:{}'.format(num_balanced[-1]))

        return self
    
    def _balanced_cov(self, X, y):
        """
        Method to compute number of balanced covariates. This should operate on all samples not on batches.
        :param X:
        :param y:
        :return: Number of balanced covariates
        """

        zeros = (y == 0).nonzero().squeeze()
        ones = y.nonzero().squeeze()

        with torch.no_grad():
            score = self.model(X).squeeze()
            weight = (y / (score + self.eps) + (1 - y) / (1 - score + self.eps)).unsqueeze(-1).repeat(1, X.shape[1])

            weight_zeros = torch.index_select(weight, 0, zeros)
            weight_ones = torch.index_select(weight, 0, ones)

            X_zeros = torch.index_select(X, 0, zeros)
            X_ones = torch.index_select(X, 0, ones)

            zeros_mean = torch.sum(weight_zeros * X_zeros, 0) / torch.sum(weight_zeros, 0)
            ones_mean = torch.sum(weight_ones * X_ones, 0) / torch.sum(weight_ones, 0)

            # Unbiased weighted variance (reliability weights)
            zeros_var = (torch.sum(weight_zeros, 0) /
                         ((torch.sum(weight_zeros, 0)) ** 2 - torch.sum(weight_zeros ** 2, 0)) *
                         torch.sum(weight_zeros * (X_zeros - zeros_mean) ** 2, 0))
            ones_var = (torch.sum(weight_ones, 0) /
                        ((torch.sum(weight_ones, 0)) ** 2 - torch.sum(weight_ones ** 2, 0)) *
                        torch.sum(weight_ones * (X_ones - ones_mean) ** 2, 0))

            # Handle calculation of norm_diff gracefully
            numer = torch.abs(zeros_mean - ones_mean)
            denom = torch.sqrt((zeros_var + ones_var) / 2)

            # Compute normalized difference where denominator is non_zero
            norm_diff = (torch.masked_select(numer, denom.ne(0)) / torch.masked_select(denom, denom.ne(0)))
            num_balanced = torch.sum(torch.le(norm_diff, self.balance_threshold)).item()

            # When denominator is zero compute cases where numerator is also zero
            num_numer_zero = torch.sum((torch.masked_select(numer, denom.eq(0))).eq(0)).item()
            num_balanced += num_numer_zero

            # When demoninator is zero and numerator is nonzero raise warning
            num_numer_nonzero = torch.sum((torch.masked_select(numer, denom.eq(0))).ne(0)).item()
            if num_numer_nonzero > 0 and self.verbose:
                self.logger.warning('Perfect separation detected for some covariates...')

        return num_balanced
    
    def _step(self, X, y, optimizer, criterion):
        """
        Method to perform one step of forward + back + update weights
        :param X:
        :param y:
        :param optimizer: Pytorch optimizer
        :param criterion: Tuple of Pytorch loss functions (BCELoss(), MSELoss())
        :return: bceloss, mseloss
        """

        zeros = (y == 0).nonzero().squeeze()
        ones = y.nonzero().squeeze()

        prop_criterion, cov_criterion = criterion

        optimizer.zero_grad()

        score = self.model(X).squeeze()

        # Propensity BCE loss
        loss_prop = prop_criterion(score, y)

        # Covariates balance loss
        weight = (y / (score + self.eps) + (1 - y) / (1 - score + self.eps)).unsqueeze(-1).repeat(1, X.shape[1])

        weight_zeros = torch.index_select(weight, 0, zeros)
        weight_ones = torch.index_select(weight, 0, ones)

        X_zeros = torch.index_select(X, 0, zeros)
        X_ones = torch.index_select(X, 0, ones)

        zeros_mean = torch.sum(weight_zeros * X_zeros, 0) / torch.sum(weight_zeros, 0)
        ones_mean = torch.sum(weight_ones * X_ones, 0) / torch.sum(weight_ones, 0)

        loss_cov = cov_criterion(zeros_mean, ones_mean)

        loss_ratio = (loss_prop / loss_cov).item()

        loss = loss_prop + self.nu * loss_ratio * loss_cov

        loss.backward()
        optimizer.step()

        return loss_prop.item(), loss_cov.item()

 
    
    def predict(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        check_is_fitted(self, attributes=['loss_stats_'])
        # Input validation
        X = check_array(X)
        X = torch.tensor(X, dtype=torch.float32, device=torch.device(self.device))
        with torch.no_grad():
            score = self.model(X).squeeze().cpu().numpy()

        labels = (score >= 0.5).astype('int')

        return labels

    def predict_proba(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        check_is_fitted(self, attributes=['loss_stats_'])
        # Input validation
        X = check_array(X)
        X = torch.tensor(X, dtype=torch.float32, device=torch.device(self.device))
        with torch.no_grad():
            score = self.model(X).squeeze().cpu().numpy()

        return np.stack([1 - score, score], axis=1)
        
       
    
