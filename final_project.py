from pathlib import Path
import matplotlib.pyplot as plt

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import pandas as pd

pd.set_option('display.max_columns', None)

"""
This data is collected by the Allen Institute. Conveniently the institute provides a SDK for easily accessing/analyzing 
the desired data, which you can read more about here: https://allensdk.readthedocs.io/en/latest/index.html

The data I chose to study was the Visual Behavior 2P Project (Optical Physiology)

Full whitepaper of experiment/data pipeline: https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/4e/be/4ebe2911-bd38-4230-86c8-01a86cfd758e/visual_behavior_2p_technical_whitepaper.pdf
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

# Confirming your allensdk version
print(f"Your allensdk version is: {allensdk.__version__}")


data_storage_directory = Path("data")

cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

# High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
all_behavior_sessions = cache.get_behavior_session_table()

# Lists of all features for a given recording session: described in detail here: https://tinyurl.com/ypmbuz4v
#print(all_behavior_sessions.columns)

'''
Important Attributes
    * session_type:    behavioral training stage or 2-photon imaging conditions for that particular recording session. e
                        e.g which types of images are shown (gratings vs natural images, etc)
    * mouse_id:        unique "name" for each mouse
    * project_codes:   dictate whether mice trained on image set A or image set B

'''



# Extract a single recording session (behavior vs ophys = 2-photon recordings)
# behavior_session = cache.get_behavior_session(behavior_session_id=870987812)




# Extracting Data for a single mouse: https://tinyurl.com/yac6d8cc
print(all_behavior_sessions['mouse_id'].unique())

mouse_id = 457841
specific_mouse_table = all_behavior_sessions[all_behavior_sessions['mouse_id'] == mouse_id]


for behavior_session in specific_mouse_table.index:
    print(hehavior)