from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, load_npz

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

pd.set_option('display.max_columns', None)

"""
This file builds our classification models off of either the PCA coordinates developed in pca.py or the full sparse
images outputted by the image_processing.py file. It does a hyperparameter search over two base models: Logistic 
Regression and Standard Vector Regression (an adaptation of the SVM). The main program at the bottom can be editted
if you wish to change the location files are saved, and the build_classifier method can be modified if you wish to
adjust any of the models or parameters used in training.
"""


def build_classifier(X, y, test_frac=0.2, results_filepath="data\\results\\classification\\classification_results.csv"):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("regressor", LogisticRegression())
    ])

    param_grid = [
        {
            "regressor": [LogisticRegression()],
            "regressor__penalty": ["elasticnet"],
            "regressor__C": [10, 100,1000],
            "regressor__max_iter": [1000],
            "regressor__l1_ratio": [0.3, 0.5, 0.7]
        },
        # {
        #     "regressor": [SVC()],
        #     "regressor__kernel": ["linear", "poly", "rbf", "sigmoid"],
        #     "regressor__C": [1e-3, 0.1, 1, 10, 100],
        #     "regressor__gamma": [1e-8, 1e-4, 0.1, 10, 100, 'auto', 'scale'],
        #     "regressor__degree": [3, 7, 10],
        # },
        {
            "regressor": [SVR()],
            "regressor__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "regressor__degree": [3, 5, 7],
            "regressor__C": [100, 1000, 1e5],
            "regressor__gamma": [1e-8, 1e-4, 'auto', 'scale'],
            #"regressor__epsilon": [1e-6, 1e-3, 1, 10, 100]
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

        {
            "regressor": [RandomForestClassifier()],
            "regressor__n_estimators": [100, 200, 400, 1000],
            "regressor__criterion": ["gini", "entropy"],
            "regressor__bootstrap": [True, False],
        },
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

    result.to_csv(Path(results_filepath), sep=",")

    print("done!")


if __name__ == "__main__":

    # gives experiment_ids for each image we processed
    processed_ids = np.genfromtxt("data\\results\\processed_image_ids.csv", dtype=int)

    # loads data matrices X (either use PCA coordinates or full images)
    pca_coordinates = np.genfromtxt("data\\results\\pca\\pca_coordinates.csv", delimiter=",")
    sparse_images = load_npz("data\\results\\sparse_images.npz")

    # Confirming your allensdk version
    print(f"Your allensdk version is: {allensdk.__version__}")

    data_storage_directory = Path("data")

    # object for downloading actual data from S3 Bucket (using cache.get_behavior_session(behavior_session_id)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

    # High-level overview of the behavior sessions in Visual Behavior Dataset (Pandas Dataframe)
    all_ophys_experiments = cache.get_ophys_experiment_table()

    # look only at experiments we used to create PCA coordinates
    all_ophys_experiments = all_ophys_experiments.loc[processed_ids]

    # Construct Labels for novelty and experience/expectation
    # +1 = familiar, 0 = novel
    familiarity = (all_ophys_experiments['experience_level'].values == "Familiar").astype(np.int32)

    # 0 = active, 1 = passive
    activity = all_ophys_experiments['session_type'].str.contains("passive").values.astype(np.int32)

    build_classifier(pca_coordinates, familiarity,
                     results_filepath="data\\results\\classification\\pca_familiarity_classification.csv")
    build_classifier(sparse_images, familiarity,
                     results_filepath="data\\results\\classification\\sparse_img_familiarity_classification.csv")
    build_classifier(pca_coordinates, activity,
                     results_filepath="data\\results\\classification\\pca_engagement_classification.csv")
    build_classifier(sparse_images, activity,
                     results_filepath="data\\results\\classification\\sparse_img_engagement_classification.csv")
