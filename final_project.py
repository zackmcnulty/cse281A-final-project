from pathlib import Path
import matplotlib.pyplot as plt
import itertools as itr
import pandas as pd
import os

from extract_mouse_metrics import extract_mouse_metrics


import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

pd.set_option('display.max_columns', None)

"""
This file serves no other role other than helping familiarize myself with the Allen institute SDK and the features
of the Visual Behavior Optical Physiology dataset.


This data is collected by the Allen Institute. Conveniently the institute provides a SDK for easily accessing/analyzing 
the desired data, which you can read more about here: https://allensdk.readthedocs.io/en/latest/index.html

The data I chose to study was the Visual Behavior 2 Photon Project (Optical Physiology)

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
    Familiar sessions are OPHYS_0, OPHYS_1, OPHYS_2, OPHYS_3 and novel are Ophys_4, OPHYS_5, OPHYS_6
    * passive = given daily water and view stimulus without lick spout (unable to earn rewards) to see how engagement 
                in the task affects the neural dynamics.
"""

# Confirming your allensdk version
print(f"Your allensdk version is: {allensdk.__version__}")

data_storage_directory = Path("data")

# object for downloading actual data from S3 Bucket (using cache.get_behavior_session(behavior_session_id)
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

# High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
all_behavior_sessions = cache.get_behavior_session_table().sort_index()
#all_ophys_sessions = cache.get_ophys_session_table().sort_values(by=['mouse_id', 'session_type'])
all_ophys_sessions = cache.get_ophys_session_table().sort_index()

# Lists of all features for a given recording session: described in detail here: https://tinyurl.com/ypmbuz4v
print(all_behavior_sessions.columns)
test = all_behavior_sessions[['mouse_id', 'session_type']]



'''
Important Attributes
    * session_type:    behavioral training stage or 2-photon imaging conditions for that particular recording session. e
                        e.g which types of images are shown (gratings vs natural images, etc)
    * mouse_id:        unique "name" for each mouse
    * project_codes:   dictate whether mice trained on image set A or image set B

'''

# Extract a single recording session (behavior vs ophys = 2-photon recordings)
behavior_session = cache.get_behavior_session(behavior_session_id=870987812)

print(list(behavior_session.get_performance_metrics().keys()))

all_session_types = all_ophys_sessions['session_type'].unique()

extract_mouse_metrics(cache, all_session_types,
                      check_point_file="data\\results\\metrics_test.csv",
                      id_file="data\\results\\id_test.csv")

cache.get_behavior_session()