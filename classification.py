from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


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

    # User-defined filepaths

    # filepath of PCA coordinates produced in pca.py
    pca_filepath = Path("data\\results\\pca\\pca_coordinates.csv")

    # filepath to processed experiment ids file produced from image_processing.py
    processed_ids_filepath = Path("data\\results\\processed_image_ids.csv")

    ########################################################################

    processed_ids = np.genfromtxt(processed_ids_filepath, dtype=int)  # gives experiment_ids for each image in pca_coordinates
    pca_coordinates = np.genfromtxt(pca_filepath, delimiter=",")      # each row is pca coordinates of 1 image

    # Confirming your allensdk version
    print(f"Your allensdk version is: {allensdk.__version__}")

    data_storage_directory = Path("data")

    # object for downloading actual data from S3 Bucket (using cache.get_behavior_session(behavior_session_id)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

    # High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
    # all_ophys_sessions = cache.get_ophys_session_table().sort_index()
    all_ophys_experiments = cache.get_ophys_experiment_table()

    # look only at experiments we used to create PCA coordinates
    all_ophys_experiments = all_ophys_experiments.loc[processed_ids]

    # +1 = familiar, 0 = novel
    y = all_ophys_experiments['experience_level'].values
    familiarity = (y == "Familiar").astype(np.int32)



    # Run Classification Algorithm ################################################################

    X_train, X_test, y_train, y_test = train_test_split(pca_coordinates, familiarity, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("regressor", LogisticRegression())
    ])

    param_grid = [
        {
            "regressor": [LogisticRegression(multi_class="ovr")],
            "regressor__penalty": ['l1', 'l2', "elasticnet", 'none'],
            "regressor__C": [1e-4, 0.1, 1, 5, 10, 20, 50, 100],
            "regressor__max_iter": [1000],
            "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        {
            "regressor": [SVC()],
            "regressor__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "regressor__C": [1e-4, 1e-2, 1, 5, 10, 20, 50],
            "regressor__gamma": [1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 'auto', 'scale'],
        },
        {
            "regressor": [SVR()],
            "regressor__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "regressor__C": [1e-4, 1e-2, 1, 5, 10, 20, 50],
            "regressor__gamma": [1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 'auto', 'scale'],
            "regressor__epsilon": [1e-6, 1e-4, 1e-2, 0.1, 1, 2, 4]
        },

        # {
        #     "regressor": [Lasso(copy_X=True)],
        #     "regressor__alpha": [1e-4, 0.01, 0.1, 0.5, 1, 2],
        #     "regressor__selection": ["cyclic", "random"],
        #
        # },

        # {
        #     "regressor": [KNeighborsClassifier()],
        #     "regressor__n_neighbors": [3, 7, 10],
        #     "regressor__weights": ["uniform", "distance"],
        # },

        # {
        #     "regressor": [RandomForestClassifier()],
        #     "regressor__n_estimators": [100, 200, 400, 1000],
        #     "regressor__criterion": ["gini", "entropy"],
        #     "regressor__bootstrap": [True, False],
        # },
    ]

    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5)
    grid.fit(X_train, y_train)

    print(f"Best score: {np.sqrt(-1.0 * grid.best_score_)}")
    print(f"Best Estimator: {grid.best_estimator_}")

    print("\n Test set results\n")

    yhat = grid.predict(X_test)
    test_score = np.sqrt(-grid.score(X_test, y_test))

    print(f"Score on test set: {test_score}")

    result = pd.DataFrame(grid.cv_results_)

    result.to_csv(Path("data\\results\\classification\\classification_results.csv"), sep=",")

    print("done!")
