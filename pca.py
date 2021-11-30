from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

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

