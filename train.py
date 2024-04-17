import os
from catboost import CatBoostClassifier
from sklift.models import SoloModel, TwoModels
from sklift.datasets import fetch_x5
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

def load_and_process_dataset(features_output, train_output):
    dataset_bundle = load_x5()
    clients_info = dataset_bundle.data["clients"].set_index("client_id")
    training_data = pd.concat(
        [dataset_bundle.data["train"], dataset_bundle.treatment, dataset_bundle.target], axis=1
    ).set_index("client_id")

    client_features = clients_info.copy()
    client_features['first_issue_time'] = (
        pd.to_datetime(client_features['first_issue_date']) - pd.Timestamp('1970-01-01')
    ) // pd.Timedelta('1s')
    client_features['first_redeem_time'] = (
        pd.to_datetime(client_features['first_redeem_date']) - pd.Timestamp('1970-01-01')
    ) // pd.Timedelta('1s')
    client_features['first_issue_redeem_delay'] = (
        client_features['first_redeem_time'] - client_features['first_issue_time']
    )
    client_features.drop(['first_issue_date', 'first_redeem_date'], axis=1, inplace=True)

    client_features.to_parquet(features_output)
    training_data.to_parquet(train_output)

def setup_model(model_type = 'solo'):
    cat_params = params = {
        "iterations": 20,
        "thread_count": 2,
        "random_state": 17,
        "silent": True,
    }
    if model_type == 'solo':
        return SoloModel(estimator=CatBoostClassifier(**cat_params))
    else:
        return TwoModels(
        estimator_trmnt=CatBoostClassifier(**cat_params),
        estimator_ctrl=CatBoostClassifier(**cat_params),
        method='vanilla'
        )

def train_given_model(model, features_path, train_path, **kwargs):
    features = pd.read_parquet(features_path)
    df_train = pd.read_parquet(train_path)
    learning_indices, _ = train_test_split(df_train.index, test_size=0.2)
    X_train = features.loc[learning_indices]
    y_train = df_train.loc[learning_indices, 'target']
    treatment_train = df_train.loc[learning_indices, 'treatment_flg']
    model.fit(X_train, y_train, treatment_train, **kwargs)
    return model

if __name__ == "__main__":
    features_output_path = 'data/processed_features.parquet'
    train_output_path = 'data/processed_train_data.parquet'
    if not os.path.exists(features_output_path):
        load_and_process_dataset(features_output_path, train_output_path)