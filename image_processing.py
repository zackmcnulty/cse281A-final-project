from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

pd.set_option('display.max_columns', None)

"""
This data is collected by the Allen Institute. Conveniently the institute provides a SDK for easily accessing/analyzing 
the desired data, which you can read more about here: https://allensdk.readthedocs.io/en/latest/index.html

The data I chose to study was the Visual Behavior 2P Project (Optical Physiology)

Full whitepaper of experiment/data pipeline: https://tinyurl.com/yc5tnnyk
Instructions for interacting with the Allen SDK: https://allensdk.readthedocs.io/en/latest/visual_behavior_optical_physiology.html

Spark Notes:
* Images (natural scenes) flashed in front of mice and they were trained to detect when the image changes and gives a response
* Data Collected:
    * Ca+ Fluorescent imaging of visual cortex 
    * Ophys = optical physiology recordings

Potential things to study:
* Mice were tested both on images seen in training and novel images. Any difference? What role does novelty play in the encoding
* Building a predictor to guess mouse's response and analyszing which factors are most heavily weighted (runs into overparameterization problem
        where we have convergence of predictions but not necessarily of weights, so analyzing weights may be misleading.
* Classify active/passive and novel/familar image sets (see figure 5 of whitepaper, pg 6) 
"""


def make_sparse(image, threshold=0.7):
    """

    :param image:         black/white image to be converted to spare matrix
    :param threshold:     cutoff threshold: zero out all pixel values below this threshold
    :return:              Scipy sparse csr_matrix of compressed image
    """

    # zero out all entries below a threshold
    sparse_image = image * (image >= threshold)

    sparsity = 100.0 * np.count_nonzero(sparse_image) / image.size
    print(f"\n Image has been reduced to {sparsity}% percent of its original size using threshold {threshold}\n")

    return csr_matrix(sparse_image)









if __name__ == "__main__":

    # Confirming your allensdk version
    print(f"Your allensdk version is: {allensdk.__version__}")

    data_storage_directory = Path("data")

    # object for downloading actual data from S3 Bucket (using cache.get_behavior_session(behavior_session_id)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

    # all_ophys_sessions = cache.get_ophys_session_table().sort_values(by=['mouse_id', 'session_type'])
    all_ophys_sessions = cache.get_ophys_session_table().sort_index()

    all_ophys_experiments = cache.get_ophys_experiment_table()
    all_ophys_experiment_ids = all_ophys_experiments.index.to_list()

    ophys_experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=all_ophys_experiment_ids[33])

    print(ophys_experiment.list_data_attributes_and_methods())

    """
    ['average_projection', 'behavior_session_id', 'cell_specimen_table', 'corrected_fluorescence_traces', 'dff_traces', 
    'events', 'eye_tracking', 'eye_tracking_rig_geometry', 'get_cell_specimen_ids', 'get_cell_specimen_indices', 
    'get_dff_traces', 'get_performance_metrics', 'get_reward_rate', 'get_rolling_performance_df', 
    'get_segmentation_mask_image', 'licks', 'max_projection', 'metadata', 'motion_correction', 'ophys_experiment_id', 
    'ophys_session_id', 'ophys_timestamps', 'raw_running_speed', 'rewards', 'roi_masks', 'running_speed', 
    'segmentation_mask_image', 'stimulus_presentations', 'stimulus_templates', 'stimulus_timestamps', 
    'task_parameters', 'trials']

    Note each experiment has a behavior_session_id AND an ophys_experiment_id because there are several ophys experiments
    per session. And an Ophys_session_id??? 

    max_projection: takes max over full recording session (e..g over time); effects more distinct...
    average_projection: takes average over full recording session (e.g over time)
    stimulus_presentations: images being shown
    """

    thresholds = [0.05, 0.1]
    fontsize=15
    plt.figure(2)

    plt.subplot(1, len(thresholds) + 1, 1)
    max_projection = ophys_experiment.max_projection.data
    plt.imshow(max_projection, cmap='gray')
    plt.title("Max Projection", fontsize=fontsize)

    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    for i, threshold in enumerate(thresholds):

        sparse_matrix = make_sparse(max_projection, threshold=threshold)

        plt.subplot(1, len(thresholds) + 1, i+2)
        plt.imshow(sparse_matrix.toarray(), cmap='gray')
        plt.title(f"Sparse ({threshold} threshold)", fontsize=fontsize)

        ax = plt.gca()
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


