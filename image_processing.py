from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback

from scipy.sparse import csr_matrix, vstack, save_npz, load_npz

from sklearn.decomposition import TruncatedSVD

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

pd.set_option('display.max_columns', None)

"""
This file handles the downloading and preprocessing of the 2-photon Ca imaging photos. It converts such photos to sparse 
matrices using thresholding as described in the project paper. The output is two new files: a sparse_images.npz file 
which stores the scipy sparse matrix for many of the ophys experiments (the max projection to be precise) where
each row corresponds to the image from a single experiment. Since some files failed to load or ran into other issues,
some experiments are missing. The processed_image_ids.csv file tracks which experiments are present in the 
sparse_images.npz. Namely, entry i of the csv gives the ophys_experiment_id of experiment which produced the image
in row i of the sparse matrix (the 512 x 512 images are flattened to vectors and stored as rows). 

For the purposes of this project, we used threshold 0.2 which reduced the image size to about 1-5% of its original size
most of the time.
"""


def make_sparse(image, threshold=0.7, flatten=False):
    """
        Converts a black and white image to a sparse matrix by zeroing out all pixels that fall below the given
        threshold (pixel values are in [0,1]). Pads image with zeros if it is not 512 x 512

    :param image:         black/white image to be converted to spare matrix
    :param threshold:     cutoff threshold: zero out all pixel values below this threshold
    :param flatten:       If true, flatten image into vector (by concatenating rows) before converting to sparse
    :return:              Scipy sparse csr_matrix of compressed image
    """

    # zero out all entries below a threshold
    sparse_image = image * (image >= threshold)

    # pad image with zeros if it is not 512 x 512
    height = sparse_image.shape[0]
    width = sparse_image.shape[1]
    if height != 512 or width != 512:
        sparse_image = np.pad(sparse_image, [(0, 512-height), (0, 512-width)])



    sparsity = 100.0 * np.count_nonzero(sparse_image) / image.size
    print(f"\nImage has been reduced to {sparsity}% percent of its original size using threshold {threshold}\n")

    if flatten:
        # flatten by concatinating rows of the 2D matrix (from top to bottom)
        sparse_image = np.reshape(sparse_image, (1, sparse_image.size), order='C')

    return csr_matrix(sparse_image)


def plot_example_of_thresholding(thresholds, ophys_experiment_id):

    """
        Plots the max projection of the given ophys experiment alongside the result from applying the thresholding
        approach at the given thresholds.

    :param thresholds:              list of thresholds to plot
    :param ophys_experiment_id:     experiment id
    """
    ophys_experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=ophys_experiment_id)
    fontsize = 15
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


def stack_all_sparse_matrices(cache, threshold=0.1, filename="data\\results\\sparse_images.npz",
                              processed_ids_file = "data\\results\\processed_image_ids.csv",
                              checkpoint_file="data\\results\\tmp\\unprocessed_image_ids.csv"):
    """

    Builds up a sparse matrix where each row is a sparsified max projection image for a single OPHYS experiment
    (created using make_sparse)


    :param cache:
    :param threshold:           see make_sparse threshold parameter
    :param filename:            npz file to store collection of sparse matrices at
    :param processed_ids_file:  csv file storing all OPHYS experiment ids processed so later we can refer to it to find
                                which row of produced sparse_matrix correspond to which experiments.
    :param checkpoint_file:     where to store list of unproccessed_ids so process can be restarted in case program fails
    """

    f = Path(filename)
    pid = Path(processed_ids_file)
    ckp = Path(checkpoint_file)

    all_ophys_experiments = cache.get_ophys_experiment_table()

    # load previous work
    if ckp.is_file() and f.is_file():
        sparse_images = load_npz(f)
        unprocessed_ids = np.genfromtxt(ckp, dtype=int)
        processed_ids = np.genfromtxt(pid, dtype=int)

        print(f"Loading previous work: sparse matrix from {filename}, processed ids from {processed_ids_file}, and "
              f"unprocessed ids from {checkpoint_file}")

    # else if no prexisting work, start from beginning
    else:

        print(f"Creating new files: sparse matrix stored at {filename}, processed ids at {processed_ids_file}, and "
              f"unprocessed ids from {checkpoint_file}")

        unprocessed_ids = all_ophys_experiments.index.to_list()

        # fenceposting
        ophys_experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=unprocessed_ids[0])

        sparse_images = make_sparse(ophys_experiment.max_projection.data, threshold=threshold, flatten=True)
        processed_ids = np.array([unprocessed_ids[0]])
        print(f'\nCompleted Ophys Experiment ID {unprocessed_ids[0]}\n')
        unprocessed_ids = unprocessed_ids[1:]

    for i, next_id in enumerate(unprocessed_ids):

        try:
            ophys_experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=next_id)
        except Exception as err:
            print(f"\nFailed to load ophys experiment id: {next_id}\n")
            traceback.print_exc()
            continue

        try:

            next_sparse_image = make_sparse(ophys_experiment.max_projection.data, threshold=threshold, flatten=True)
            sparse_images = vstack((sparse_images, next_sparse_image))
        except Exception as err:
            # common error: some images are 512 x 451 instead of 512 x 512, so they have incompatible sizes with
            # previously processed images
            print(f"\nFailed to stack sparse image id {next_id}\n")
            traceback.print_exc()
            continue

        processed_ids = np.append(processed_ids, [next_id])
        print(f'\nCompleted Ophys Experiment ID {next_id}\n')

        save_npz(f, sparse_images)
        np.savetxt(pid, processed_ids)
        if i + 1 < len(unprocessed_ids):
            np.savetxt(ckp, unprocessed_ids[i+1:])
        else:
            print("\nAll ids have been processed\n")


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

    #print(ophys_experiment.list_data_attributes_and_methods())

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

    thresholds = []
    experiment_id = all_ophys_experiment_ids[27]

    # plots an example of thresholding used in this paper
    plot_example_of_thresholding(thresholds, experiment_id)

    # Sparisfy all max projection images, flatten each to a row vector, and collect all in a single sparse matrix
    # stack_all_sparse_matrices(cache, threshold=0.2)

    print("hello dad")
