
# Predictive Model for Personal Training Sales (Recommendation System)

This project implements a predictive model to forecast potential personal training sales based on user interactions and demographics. The model utilizes machine learning techniques, including logistic regression and random forest classification.

## Overview

The primary goal of this project is to predict whether a subscriber will purchase personal training services in the upcoming week. The model takes into account user interactions such as app interactions, scheduled appointments, SMS sent, and chat messages. Additionally, demographic information like gender and age is considered.

## Data

The project uses two main datasets:
- `events_data.csv`: Records user interactions and events.
- `subscribers_data.csv`: Contains subscriber information, including gender, age, and the effective date of their subscription.

## How the Code Works

The code is structured as follows:

- **Data Loading and Preprocessing:** The `load_and_preprocess_data` function reads events and subscribers data, handles missing values, converts date fields, and merges relevant information.

- **Filtering Events by Date:** The `filter_events_by_date` function filters events data based on a specific date and duration, capturing events both backward and forward in time.

- **Creating Result DataFrame:** The `create_result_df` function generates a DataFrame with event type counts for each member, creating features for the predictive model.

- **Training and Evaluating Models:** The project supports two classification models: logistic regression and random forest. Functions like `train_and_evaluate_model_logistic_regression` and `train_and_evaluate_random_forest` train the models, evaluate their performance, and plot diagnostic curves.

- **Hyperparameter Tuning:** The code includes hyperparameter tuning for the random forest model using RandomizedSearchCV. You can experiment with different hyperparameter values to optimize model performance.

- **Making Predictions:** The `make_predictions` function generates predictions for subscribers who have not purchased personal training services. It ranks and recommends the top 10 individuals.

- **Results and Evaluation:** The `collect_results` function collects and prints key results, including model performance metrics and the success of top 10 predictions compared to actual outcomes.


