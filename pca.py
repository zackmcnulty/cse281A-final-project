from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, load_npz

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

from sklearn.decomposition import TruncatedSVD # designed to work efficiently on sparse matrices

pd.set_option('display.max_columns', None)

"""
This file applies PCA to the sparse representations of the Ca imaging images produced in image_processing.py in 
order to obtain simplified coordinates for our neural representation (e.g. by taking the first few principal components

Relevant stackoverflow:
https://stackoverflow.com/questions/33603787/performing-pca-on-large-sparse-matrix-by-using-sklearn
https://stackoverflow.com/questions/10718455/apply-pca-on-very-large-sparse-matrix

TruncatedPCA
"""


if __name__ == "__main__":

    # Confirming your allensdk version
    print(f"Your allensdk version is: {allensdk.__version__}")

    data_storage_directory = Path("data")

    # object for downloading actual data from S3 Bucket (using cache.get_behavior_session(behavior_session_id)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

    # High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
    all_ophys_sessions = cache.get_ophys_session_table().sort_index()

    sparse_filepath = Path("data\\results\\sparse_images.npz")
    processed_ids_filepath = Path("data\\results\\processed_image_ids.csv")

    # images are stored as rows of the following sparse matrix
    sparse_images = load_npz(sparse_filepath)
    processed_ids = np.genfromtxt(processed_ids_filepath)

    num_images = len(processed_ids)

    # plot one sparse image to check sparse_images were stored correctly
    # index = 10
    # plt.subplot(121)
    # plt.imshow(sparse_images[index, :].toarray().reshape((512, 512)), cmap='gray')
    #
    # plt.subplot(122)
    # test_experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=processed_ids[index])
    # plt.imshow(test_experiment.max_projection.data, cmap='gray')
    # plt.show()

    # for i in range(num_images):
    #     plt.imshow(sparse_images[i, :].toarray().reshape(512, 512), cmap='gray')
    #     plt.show()

    n_components = 750 # number of SVD components to calculate

    svd_model = TruncatedSVD(n_components=n_components)
    svd_model.fit(sparse_images)

    print(np.cumsum(svd_model.explained_variance_ratio_))

    # for i in range(n_components):
    #     plt.imshow(svd_model.components_[i, :].reshape(512, 512), cmap='gray')
    #     plt.show()


    # images transformed into SVD basis
    svd_images = svd_model.transform(sparse_images)
    for i in range(num_images):
        svd_coordinates = svd_images[i, :]

        low_dim_image = svd_coordinates @ svd_model.components_
        plt.imshow(low_dim_image.reshape((512, 512)), cmap='gray')
        plt.show()





