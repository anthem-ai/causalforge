import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_ite_distribution(ites):
    ate = np.mean(ites)
    sns.set_style("darkgrid")
    g = sns.displot(pd.DataFrame(ites,columns=['ITE']), 
                    x="ITE", kind="kde", 
                    fill=True, 
                    common_norm=False)
    g.fig.set_size_inches(10,7)
    g.set_axis_labels("ITE","Density")
    plt.axvline(ate, color='red')
    plt.text(ate,0,'ATE',rotation=0,color='red')
    return g 