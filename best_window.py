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
