import pandas as pd
import numpy as np
from pathlib import Path

"""
This program queries the Allen Institute Database to collect performance metrics for each behavior session, including 
those with Optical Physiology. These metrics capture how well the given mouse in the session was able to perform
the behavior task: recognizing when the presented image changed. It counts true/false positive/negatives, as well as 
a measure of classification accuracy called D prime which balances true positives and false positives. This metric is
described in more detail in the whitepaper.

            'trial_count', 'go_trial_count', 'catch_trial_count',
           'hit_trial_count', 'miss_trial_count', 'false_alarm_trial_count',
           'correct_reject_trial_count', 'auto_reward_count', 'earned_reward_count',
           'total_reward_count', 'total_reward_volume', 'maximum_reward_rate',
           'engaged_trial_count', 'mean_hit_rate', 'mean_hit_rate_uncorrected',
           'mean_hit_rate_engaged', 'mean_false_alarm_rate', 'mean_false_alarm_rate_uncorrected',
           'mean_false_alarm_rate_engaged', 'mean_dprime', 'mean_dprime_engaged',
           'max_dprime', 'max_dprime_engaged'

"""



# Extracting Data for a single mouse: https://tinyurl.com/yac6d8cc
def extract_mouse_metrics(cache, all_session_types, check_point_file="data\\results\\mouse_TPR_FPR.csv", id_file="data\\results\\remaining_ids.csv"):

    """
        Extracts summary performance metrics for each session and organizes them into a single dataframe


    :param cache:               VisualBehaviorOphys Cache object for collecting data from Allen S3 bucket
    :param all_session_types:   Types of sessions to collect TPR and FPR for: cache.get_behavior_session_table()['session_type'] are valid options
    :param check_point_file:    Checkpoint file for saving partial work (as this calculation takes a long time);
                                Last row should be the mouse_ids that have been processed already
    :param id_file              stores remaining behavior_session_ids left to process
    :return:                    Dataframe containing performance metrics
    """

    performance_metrics = ['trial_count', 'go_trial_count', 'catch_trial_count',
                           'hit_trial_count', 'miss_trial_count', 'false_alarm_trial_count',
                           'correct_reject_trial_count', 'auto_reward_count', 'earned_reward_count',
                           'total_reward_count', 'total_reward_volume', 'maximum_reward_rate',
                           'engaged_trial_count', 'mean_hit_rate', 'mean_hit_rate_uncorrected',
                           'mean_hit_rate_engaged', 'mean_false_alarm_rate', 'mean_false_alarm_rate_uncorrected',
                           'mean_false_alarm_rate_engaged', 'mean_dprime', 'mean_dprime_engaged',
                           'max_dprime', 'max_dprime_engaged']

    check_point_file = Path(check_point_file)
    id_file = Path(id_file)

    # High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
    all_behavior_sessions = cache.get_behavior_session_table()

    # load previous results from file; skip past already computed mouse ids
    if check_point_file.is_file() and id_file.is_file():
        print(f"\nLoading previous work from {check_point_file}\n\n")

        metric_table = pd.read_csv(check_point_file)

        # load all behavior_ids left to process
        all_behavior_session_ids = np.genfromtxt(id_file)

    else:
        print(f'\nCreating new checkpointing files\n\n')

        # filter out all ideas not corresponding to the desired session types
        all_behavior_session_ids = all_behavior_sessions.query('session_type in @all_session_types').index.to_list()

        metric_table = pd.DataFrame(columns=['behavior_session_id', 'mouse_id', 'session_type'] + performance_metrics)

    metric_table.set_index(['behavior_session_id'], inplace=True)

    for i, session_id in enumerate(all_behavior_session_ids):

        mouse_id, session_type = all_behavior_sessions.loc[session_id][['mouse_id', 'session_type']]

        try:
            session = cache.get_behavior_session(session_id)
        except OSError as err:
            print(err)
            print(f"Failed to load behavior session {session_id}")
            continue

        data = session.get_performance_metrics()
        data['mouse_id'] = mouse_id
        data['session_type'] = session_type

        data = {key: [data[key]] for key in data}
        idx = pd.Index([session_id], name="behavior_session_id")

        new_row = pd.DataFrame(data=data, index=idx)
        metric_table = metric_table.append(new_row, ignore_index=False)

        """
            For each session type, find all behavior session ids for this specific mouse with that session type, and download
            the corresponding dataset for those sessions from the cache. Each dataset contains the following attributes:
             'behavior_session_id'               : key identifying this session
             'get_performance_metrics'           : summary statistics of the given session
             'get_reward_rate'                   :
             'get_rolling_performance_df'        : how well the mouse is performing over time (dprime is a measure of how well they are discriminating changes : https://tinyurl.com/3949v9ns )
             'licks'                             : timestamps/frames on when licks occurred
             'metadata'   
             'raw_running_speed'                 :
             'rewards'                           :  when rewards were administered
             'running_speed',
             'stimulus_presentations'            : one entry for each distinct stimulus, including onset/offset time/frame
             'stimulus_templates'                : what different images were shown
             'stimulus_timestamps'               : when different images were shown
             'task_parameters'                   :
             'trials'                            : view all attributes of a trial (e.g. a timeslot of an image being shown, whether or not it changed from the previous)
        """

        print(f'Completed behavior session {session_id}')
        metric_table.to_csv(check_point_file, index=True)  # save work to csv for checkpointing

        try:
            remaining_session_ids = all_behavior_session_ids[i+1:]
            np.savetxt(id_file, remaining_session_ids)
        except:
            print("Completed all sessions!")







if __name__ == "__main__":
    import allensdk
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

    # Confirming your allensdk version
    print(f"Your allensdk version is: {allensdk.__version__}")

    # Directory to store data downloaded through the AllenSDK
    data_storage_directory = Path("data")

    # object for downloading actual data from S3 Bucket (using cache.get_behavior_session(behavior_session_id)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

    # High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
    # Lists of all features for a given recording session: described in detail here: https://tinyurl.com/ypmbuz4v
    all_behavior_sessions = cache.get_behavior_session_table().sort_index()

    # all_ophys_sessions = cache.get_ophys_session_table().sort_values(by=['mouse_id', 'session_type'])
    all_ophys_sessions = cache.get_ophys_session_table().sort_index()

    all_session_types = all_ophys_sessions['session_type'].unique()

    extract_mouse_metrics(cache, all_session_types,
                          check_point_file="data\\results\\metrics_test.csv",
                          id_file="data\\results\\id_test.csv")

