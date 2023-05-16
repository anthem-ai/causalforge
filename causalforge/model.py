from abc import ABC, abstractmethod
from enum import Enum



def dynamic_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class PROBLEM_TYPE(Enum):
    CAUSAL_TREATMENT_EFFECT_ESTIMATION = "causal_treatment_effect_estimation"
    PROPENSITY_ESTIMATION = "propensity_estimation"
    SYNTHETIC_DATA_GENERATION = "syntethic_data_generation"
    

class Propensity_Estimator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def build(self,params):
        pass 
    
    @abstractmethod
    def fit(self, X, treatment):
        """
        Fits the model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
        """
        pass 
    
    @abstractmethod
    def predict(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        pass 
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        pass 


class Model(ABC):
    
    def __init__(self):
        pass

    @staticmethod
    def create_model(name,
                     params,
                     problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                     multiple_treatments=False):
        if  problem_type==PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION:
            if multiple_treatments:
                raise Exception("Multiple treatments not supported yet")
            else:
                if name == 'bcauss':
                    klass = dynamic_import('causalforge.models.BCAUSS')
                elif name == 'dragonnet':
                    klass = dynamic_import('causalforge.models.DragonNet')
                elif name == 'ganite':
                    klass = dynamic_import('causalforge.models.Ganite')
                elif name == 'bcaus_dr':
                    klass = dynamic_import('causalforge.models.BCAUS_DR') 
                else:
                    raise Exception("Model not supported yet::"+str(name))
                #
                net = klass()
                net.build(params)
                return net
        elif problem_type==PROBLEM_TYPE.PROPENSITY_ESTIMATION:
            if name == 'bcaus':
                klass = dynamic_import('causalforge.models.BCAUS')
            else:
                raise Exception("Model not supported yet::"+str(name))
            net = klass()
            net.build(params)
            return net
        elif problem_type==PROBLEM_TYPE.SYNTHETIC_DATA_GENERATION:
            raise Exception("problem_type not supported yet::"+str(problem_type))
        else:
            raise Exception("Invalid problem_type ::"+str(problem_type))
    
    @abstractmethod
    def build(self,params):
        pass 
    
    
    @abstractmethod
    def fit(self, X, treatment, y):
        """
        Fits the model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        pass 
    
    @abstractmethod
    def support_ite(self):
        """
        Whether the model supports individual treatment effect ("ITE") estimation 

        Returns:
            (Boolean): Whether the model supports ITE estimation
        """
        pass 
    
    @abstractmethod
    def predict_ite(self, X):
        """
        Predicts the individual treatment effect ("ITE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.array): treatment effect vector
        """
        pass 
    
    @abstractmethod
    def predict_ate(self, X,treatment,y):
        """
        Predicts the average treatment effect ("ATE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        Returns:
            (np.array): treatment effect vector
        """
        pass 

