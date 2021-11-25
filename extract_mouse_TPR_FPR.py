import itertools as itr
import pandas as pd
import numpy as np
from pathlib import Path

# Extracting Data for a single mouse: https://tinyurl.com/yac6d8cc
def extract_mouse_tpr_fpr(cache, all_session_types, ophys_only=True, check_point_file="data\\results\\mouse_TPR_FPR.csv", mice_file="data\\results\\unprocessed_mice.csv"):

    """

    :param cache:               VisualBehaviorOphys Cache object for collecting data from Allen S3 bucket
    :param all_session_types:   Types of sessions to collect TPR and FPR for: cache.get_behavior_session_table()['session_type'] are valid options
    :param check_point_file:    Checkpoint file for saving partial work (as this calculation takes a long time);
                                Last row should be the mouse_ids that have been processed already
    :param ophys_only:          Search through only OPHYS experiments (if True) vs all behavior sessions (if false)
    :param mice_file:           Temporary file used for checkpoint to store already processed list of mice
    :return:
    """

    check_point_file = Path(check_point_file)
    mice_file = Path(mice_file)

    if ophys_only:
        all_behavior_sessions = cache.get_ophys_session_table()
    else:
        # High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
        all_behavior_sessions = cache.get_behavior_session_table()


    if check_point_file.is_file() and mice_file.is_file():  # load previous results from file; skip past already computed mouse ids
        print(f"\nLoading previous work from {check_point_file} and {mice_file}\n\n")

        accuracy_table = pd.read_csv(check_point_file)

        # skip over all mouse_ids that have already been computed
        all_mice_ids = np.genfromtxt(mice_file, dtype=int)

    else:
        print(f'\nCreating new checkpointing files\n\n')
        all_mice_ids = all_behavior_sessions['mouse_id'].unique()
        accuracy_table = pd.DataFrame(data=itr.product(all_mice_ids, all_session_types, [None], [None]),
                                      columns=['mouse_id', 'session_type', 'TPR', 'FPR'])


    accuracy_table.set_index(['mouse_id', 'session_type'], inplace=True)

    for i, mouse_id in enumerate(all_mice_ids):
        specific_mouse_table = all_behavior_sessions[all_behavior_sessions['mouse_id'] == mouse_id]

        for session_type in all_session_types:

            # Collect all data for this given mouse under the specified session type
            # extract all ids for this session type
            # all_behavior_session_ids = specific_mouse_table.query(f'session_type.str.contains("{session_type}")').index.to_list()
            # TODO: apparently there are different behavior session_ids and ophys session ids
            if ophys_only:
                all_behavior_session_ids = specific_mouse_table[specific_mouse_table['session_type']
                                                                .str.contains(session_type)]['behavior_session_id']\
                                                                .to_list()

            else:
                all_behavior_session_ids = specific_mouse_table[specific_mouse_table['session_type']
                                                                .str.contains(session_type)].index.to_list()

            go_trials = 0
            catch_trials = 0
            hit_trials = 0
            false_alarm_trials = 0

            for session_id in all_behavior_session_ids:
                try:
                    data = cache.get_behavior_session(session_id).get_performance_metrics()

                except OSError as err:
                    error_msg = f"\nERROR: cannot read file associated to behavior session id : {session_id}"
                    print(error_msg)
                    print(err + "\n")
                    continue

                go_trials += data['go_trial_count']
                catch_trials += data['catch_trial_count']
                hit_trials += data['hit_trial_count']
                false_alarm_trials += data['false_alarm_trial_count']

            if go_trials > 0:
                accuracy_table.loc[(mouse_id, session_type), 'TPR'] = hit_trials / go_trials

            if catch_trials > 0:
                accuracy_table.loc[(mouse_id, session_type), 'FPR'] = false_alarm_trials / catch_trials


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

            print(f"Completed mouse ID {mouse_id} session type {session_type}")

        accuracy_table.to_csv(check_point_file, index=True)  # save work to csv for checkpointing

        try:
            remaining_mice = all_mice_ids[i+1:]
            np.savetxt(mice_file, remaining_mice)
        except:
            print("Completed all mice!")