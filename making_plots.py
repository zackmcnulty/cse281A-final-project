import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from pathlib import Path
import seaborn as sn
import numpy as np
import time

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache




"""
This file is just used to construct summary plots (to be used in the final report) out of the results produced by
the other files.

"""


# PCA Plots #######################################################
pca_filepath = Path("data\\results\\pca\\pca_coordinates.csv")
processed_ids_filepath = Path("data\\results\\processed_image_ids.csv")

fontsize=17

processed_ids = np.genfromtxt(processed_ids_filepath,
                              dtype=int)  # gives experiment_ids for each image in pca_coordinates
pca_coordinates = np.genfromtxt(pca_filepath, delimiter=",")  # each row is pca coordinates of 1 image

data_storage_directory = Path("data")

cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)
all_ophys_experiments = cache.get_ophys_experiment_table()
all_ophys_experiments = all_ophys_experiments.loc[processed_ids]

# +1 = familiar, 0 = novel
y = all_ophys_experiments['experience_level'].values

pca_coordinates = np.genfromtxt(Path("data\\results\\pca\\pca_coordinates.csv"), delimiter=",")
familiar_coordinates = pca_coordinates[y == "Familiar", :]
novel_coordinates = pca_coordinates[y != "Familiar", :]


plt.figure(2)

ax = plt.axes(projection="3d")
ax.scatter3D(familiar_coordinates[:, 0], familiar_coordinates[:, 1], familiar_coordinates[:, 2],
             label="familiar",
             color="orange",
             alpha=0.5)
ax.scatter3D(novel_coordinates[:, 0], novel_coordinates[:, 1], novel_coordinates[:, 2],
             label="novel",
             color="blue",
             alpha=0.5)
plt.legend(fontsize=fontsize)
plt.title("Distribution Components 1-3", fontsize=fontsize, fontweight='bold')
ax.set_xlabel('1st Component', fontweight='bold', fontsize=fontsize)
ax.set_ylabel('2nd Component', fontweight='bold', fontsize=fontsize)
ax.set_zlabel('3rd Component', fontweight='bold', fontsize=fontsize)


plt.show()

plt.figure(3)

ax = plt.axes(projection="3d")
ax.scatter3D(familiar_coordinates[:, 3], familiar_coordinates[:, 4], familiar_coordinates[:, 5],
             label="familiar",
             color="orange",
             alpha=0.5)
ax.scatter3D(novel_coordinates[:, 3], novel_coordinates[:, 4], novel_coordinates[:, 5],
             label="novel",
             color="blue",
             alpha=0.5)
plt.legend(fontsize=fontsize)
plt.title("Distribution Components 4-6", fontsize=fontsize, fontweight='bold')
ax.set_xlabel('4th Component', fontweight ='bold', fontsize=fontsize)
ax.set_ylabel('5th Component', fontweight ='bold', fontsize=fontsize)
ax.set_zlabel('6th Component', fontweight ='bold', fontsize=fontsize)

plt.show()

plt.figure(4)

ax = plt.axes(projection="3d")
ax.scatter3D(familiar_coordinates[:, 6], familiar_coordinates[:, 7], familiar_coordinates[:, 8],
             label="familiar",
             color="orange",
             alpha=0.5)
ax.scatter3D(novel_coordinates[:, 6], novel_coordinates[:, 7], novel_coordinates[:, 8],
             label="novel",
             color="blue",
             alpha=0.5)
plt.legend(fontsize=fontsize)
plt.title("Distribution Components 7-9", fontsize=fontsize, fontweight='bold')
ax.set_xlabel('7th Component', fontweight ='bold', fontsize=fontsize)
ax.set_ylabel('8th Component', fontweight ='bold',fontsize=fontsize)
ax.set_zlabel('9th Component', fontweight ='bold', fontsize=fontsize)

plt.show()


# HBayes Plots ####################################

sn.set_theme()
sn.set_context("talk")

filepath = "data\\results\\hbayes\\"

familiar_filepath = Path(filepath + "familiar_tpr_fpr.csv")
novel_filepath = Path(filepath + "novel_tpr_fpr.csv")

familiar_table = pd.read_csv(familiar_filepath, index_col=0).dropna()
familiar_table.index.name = "mouse_id"

novel_table = pd.read_csv(novel_filepath, index_col=0).dropna()
novel_table.index.name = "mouse_id"

num_bins = 13  # number histogram bins

plt.figure(1)
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





