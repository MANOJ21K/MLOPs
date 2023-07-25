#importing the libraries

import pandas as pd
import numpy as np
import datetime
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow
import boto3
from prefect import flow, task



@task(retries=3, retry_delay_seconds=2)
def read_data():
    dfs = []
    for i in range(1, 3):  # using just 2 months now
        data = pd.read_parquet(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-{str(i).zfill(2)}.parquet")
        dfs.append(data)

    data = pd.DataFrame()

    # Iterate over the list of DataFrames
    for df in dfs:
        data = pd.concat([data, df], ignore_index=True)

    return df



@task
def preprocessing(data):
    #lets use the 237 locationID since its the heavy volume place 
    data = data[data['PULocationID']==237]

    #taking the date and the number of trips per day
    data['date'] = data['tpep_pickup_datetime'].dt.date

    Final_data = pd.DataFrame(data.groupby('date')['VendorID'].count())

    Final_data.reset_index(inplace=True)

    Final_data.columns = ['Date', 'No of trips']

    return Final_data


@task
def window_size(Final_data):
    mlflow.set_experiment("best_window_size_237")
    for window_size in range(1,20):
        with mlflow.start_run():
            # Create empty lists to store the input and output data
            X = []
            y = []
            for i in range(window_size, len(Final_data)):
                # Extract the 12-day sequence as input
                input_sequence = Final_data['No of trips'].iloc[i-window_size:i].tolist()
                
                # Extract the 13th day count as output
                output_value = Final_data['No of trips'].iloc[i]
                
                # Append the input and output to the respective lists
                X.append(input_sequence)
                y.append(output_value)

            # Create a new DataFrame from the input and output lists
            X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(window_size)])
            y_df = pd.DataFrame({'y': y})
            new_df = pd.concat([X_df, y_df], axis=1)


            X = new_df.iloc[:, :-1]
            y = new_df['y']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Convert the data into DMatrix format for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Set the parameters for the XGBoost model
            params = {
                'objective': 'reg:squarederror',  # Use squared error as the objective for regression
                'eval_metric': 'rmse'  # Root Mean Squared Error (RMSE) as the evaluation metric
            }

            # Train the XGBoost model
            num_rounds = 100  # Number of boosting rounds
            model = xgb.train(params, dtrain, num_rounds)

            # Make predictions on the test set
            y_pred = model.predict(dtest)

            # Calculate RMSE on the test set
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.log_metric('Window_size', f"{window_size}")
            mlflow.log_metric("rmse", rmse)
    mlflow.end_run()



@task
def best_window_size(Final_data):
    # Create empty lists to store the input and output data
    X = []
    y = []
    for i in range(7, len(Final_data)):
        # Extract the 7-day sequence as input
        input_sequence = Final_data['No of trips'].iloc[i-7:i].tolist()
        
        # Extract the 8th day count as output
        output_value = Final_data['No of trips'].iloc[i]
        
        # Append the input and output to the respective lists
        X.append(input_sequence)
        y.append(output_value)

    # Create a new DataFrame from the input and output lists
    X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(7)])
    y_df = pd.DataFrame({'y': y})
    new_df = pd.concat([X_df, y_df], axis=1)


    X = new_df.iloc[:, :-1]
    y = new_df['y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    return X_train, X_test, y_train, y_test


@task
def training_model(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("best_model_237")
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_test, label=y_test)
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=10,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}
    
    search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )
    mlflow.end_run()


@task(log_prints=True)
def best_model(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("Final_Model_Total_trips_prediction_237")
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_test, label=y_test)
        best_params = {
            'learning_rate': 0.3186829398830741,
            'max_depth': 29,
            'min_child_weight': 6.747015358253847,
            'objective': 'reg:linear',
            'reg_alpha': 0.03259110542981609,
            'reg_lambda': 0.04013136677983584,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


@flow
def main_flow():
    # MLflow settings
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Load
    data = read_data()

    # Transform
    Final_data = preprocessing(data)
    print("Final data extracted")
    
    #logging the best window size
    window_size(Final_data)
    print("best window logged")

    # Train and test data post best window size
    X_train, X_test, y_train, y_test = best_window_size(Final_data)
    print("splitted the data based on best window")

    #model selection 
    training_model(X_train, X_test, y_train, y_test)
    print("best hyper parameter logged")

    #building the best model
    best_model(X_train, X_test, y_train, y_test)
    print("trained the best model")

if __name__ == "__main__":
    main_flow()