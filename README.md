# UC Berkeley CSE 281A Final Project: The Influence of Novelty and Engagement on Neural Dynamics

## Data

The data for this project was collected by the [Allen Institute](https://alleninstitute.org/). Conveniently, the institute provides its own [SDK](https://allensdk.readthedocs.io/en/latest/index.html) for easily accessing/analyzing with this data. In particular, we focused on the [Visual Behavior 2-Photon Project (Optical Physiology)](https://portal.brain-map.org/explore/circuits/visual-behavior-2p) which studies the influence of novelty and expectation on neural dynamics. Neural dynamics are measured using Calcium Imaging and this was the main datasource we used for the project.

For more details on the experiment design and data-collection process, check out the project's [whitepaper](https://tinyurl.com/yc5tnnyk).

For more information and examples on interacting with the Allen SDK, see [here](https://allensdk.readthedocs.io/en/latest/visual_behavior_optical_physiology.html)


## File Descriptions


#### extracting_mouse_metrics.py

This program queries the Allen Institute Database to collect performance metrics for each behavior session, including 
those with Optical Physiology. These metrics capture how well the given mouse in the session was able to perform
the behavior task: recognizing when the presented image changed. It counts true/false positive/negatives, as well as 
a measure of classification accuracy called D prime which balances true positives and false positives. This metric is
described in more detail in the whitepaper. Here a more complete list of the available metrics that are collected in the final table.

```
            'trial_count', 'go_trial_count', 'catch_trial_count',
           'hit_trial_count', 'miss_trial_count', 'false_alarm_trial_count',
           'correct_reject_trial_count', 'auto_reward_count', 'earned_reward_count',
           'total_reward_count', 'total_reward_volume', 'maximum_reward_rate',
           'engaged_trial_count', 'mean_hit_rate', 'mean_hit_rate_uncorrected',
           'mean_hit_rate_engaged', 'mean_false_alarm_rate', 'mean_false_alarm_rate_uncorrected',
           'mean_false_alarm_rate_engaged', 'mean_dprime', 'mean_dprime_engaged',
           'max_dprime', 'max_dprime_engaged'
```


#### hierachical_bayes.py
This code is used to run the Hierarchical Bayesian analysis of the TPR/FPR of individual mice in the behavior task
as described in the paper. It uses the output of the "extract_mouse_metrics.py" program, a csv `metrics.csv` of the performance
metrics for each individual mouse in each experiment, to perform the analysis. It outputs the MLE estimates of $\alpha, \beta$
and appends columns the estimates of the TPR and FPR estimates to the `metrics.csv` to a pair of new csvs: `hbayes/familiar_tpr_fpr.csv` and `hbayes/tpr/fpr.csv`.

#### image_processing.py
This file handles the downloading and preprocessing of the 2-photon Ca imaging photos. It converts such photos to sparse 
matrices using thresholding as described in the project paper. The output is two new files: a sparse_images.npz file 
which stores the scipy sparse matrix for many of the ophys experiments (the max projection to be precise) where
each row corresponds to the image from a single experiment. Since some files failed to load or ran into other issues,
some experiments are missing. The processed_image_ids.csv file tracks which experiments are present in the 
sparse_images.npz. Namely, entry i of the csv gives the ophys_experiment_id of experiment which produced the image
in row i of the sparse matrix (the 512 x 512 images are flattened to vectors and stored as rows). 

For the purposes of this project, we used threshold 0.2 which reduced the image size to about 1-5% of its original size
most of the time.


#### pca.py




#### classification.py



#### making_plots.py


