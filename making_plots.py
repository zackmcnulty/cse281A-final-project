import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sn

sn.set_theme()
sn.set_context("talk")


"""
This file is just used to construct summary plots (to be used in the final report) out of the results produced by
the other files.

"""

# HBayes Plots ###########################

filepath = "data\\results\\hbayes\\"

familiar_filepath = Path(filepath + "familiar_tpr_fpr.csv")
novel_filepath = Path(filepath + "novel_tpr_fpr.csv")

familiar_table = pd.read_csv(familiar_filepath, index_col=0).dropna()
familiar_table.index.name = "mouse_id"

novel_table = pd.read_csv(novel_filepath, index_col=0).dropna()
novel_table.index.name = "mouse_id"

num_bins = 13  # number histogram bins

plt.figure()
plt.subplot(131)
plt.hist(familiar_table['bayes_TPR'].to_numpy(), density=True, bins=num_bins, alpha=0.5, label="familiar", color='blue')
plt.hist(novel_table['bayes_TPR'].to_numpy(), density=True, bins=num_bins, alpha=0.5, label="novel", color='orange')
plt.legend()
plt.title("Hierarchical Bayes TPR")
plt.ylabel('density')
plt.xlabel("TPR")

plt.subplot(132)
plt.hist(familiar_table['bayes_FPR'].to_numpy(), density=True, bins=num_bins, alpha=0.5, label="familiar", color='blue')
plt.hist(novel_table['bayes_FPR'].to_numpy(), density=True, bins=num_bins, alpha=0.5, label="novel", color='orange')
plt.legend()
plt.title("Hierarchical Bayes FPR")
plt.xlabel("FPR")

plt.subplot(133)
plt.scatter(familiar_table['bayes_FPR'].to_numpy(), familiar_table['bayes_TPR'], color='royalblue', marker='.', label="familiar")
plt.scatter(novel_table['bayes_FPR'].to_numpy(), novel_table['bayes_TPR'], color='orange', marker=".", label="novel")
plt.plot([0, 1], [0,1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Partial ROC Curve')
plt.legend()

plt.show()





