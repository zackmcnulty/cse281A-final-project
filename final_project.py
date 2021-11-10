from pathlib import Path
import matplotlib.pyplot as plt

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# Confirming your allensdk version
print(f"Your allensdk version is: {allensdk.__version__}")


data_storage_directory = Path("data")

cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

# High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
behavior_sessions = cache.get_behavior_session_table()