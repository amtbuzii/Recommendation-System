# Vi labs - Data Scientist - Home Assignment

# Predictive Model for Personal Training Sales (Recommendation System)

This repository contains a Python code that implements a predictive model for personal training (PT) sales based on member activities. The script serves as a recommendation system utilizing machine learning techniques to train a logistic regression model and evaluate its performance.

## Overview

The script performs the following key steps:

## 1. Data Loading and Preprocessing:
   - Loads events and subscribers data from CSV files.
   - Drops unnecessary columns and handles missing values.
   - Converts date fields to datetime format and sorts data by date.

## 2. Filtering Events:
   - Filters events data based on a specific date and duration.
   - Divides the filtered data into two sets: backward (3 months) and forward (week) from the specific date.

## 3. Creating features:
   - Constructs a DataFrame with counts of various event types for each member.
   - Includes a binary column indicating whether a member had a PT sale event in the next week. (label)

## 4. Feature Engineering:
   - Utilizes various event types (e.g., app_interaction, personal_appointment_scheduled, human_communication_event) as features.
   - Each feature represents the count of occurrences for the corresponding event type for a specific member (In last 3 months).

## 5. Training and Evaluating Logistic Regression Model:
   - Splits the data into training and testing sets based on member IDs. (prevent overfitting)
   - Normalizes the data using Min-Max scaling. (only in the logistic regression model)
   - Trains a logistic regression model and evaluates its performance using classification metrics.

## 6. Making Predictions:
   - Predicts PT sales for individuals who have not taken PT and have an active subscription.
   - Ranks and recommends the top 10 individuals based on predicted probabilities.

## 7. Evaluating Model Success:
   - Checks the success of the model by comparing predicted top 10 individuals with actual PT sales.


## Feature Details

The script leverages a diverse set of features derived from member activities. These features include:

- **App Interaction Count:** Counts of interactions with the mobile application.
- **Personal Appointment Scheduled Count:** Counts of scheduled personal appointments.
- **Human Communication Event Count:** Counts of events involving human communication.
- **SMS Sent Count:** Counts of sent SMS messages.
- **Chat Message Sent Count:** Counts of sent chat messages.
- **Manual Email Sent Count:** Counts of manually sent emails.
- **Automated Email Sent Count:** Counts of automated (system-generated) email events.
- **Fitness Consultation Count:** Counts of fitness consultation events.
- **Usage Count:** Counts of general usage events.
- **Usage Count per month (1st, 2nd, 3rd):** Indicates whether there is any progress in the first, second, and third months.
- **Minimum Interval:** Minimum time interval between consecutive usage events
- **Maximum Interval:** Maximum time interval between consecutive usage events.

## Usage

To use the script:

1. Ensure you have the required Python libraries installed (e.g., matplotlib, pandas, sklearn).
2. Place your events and subscribers data CSV files in the 'data' directory.
3. Adjust the script parameters such as `start_date_for_train` and `date_to_predict`.
4. Run the script using `vi_final.py`.

## Results

The script prints relevant metrics such as accuracy, precision, recall, F1 score, and ROC-AUC for the logistic regression model. It also provides precision@k and recall@k for the top 10 predictions.

**Note:** The script currently focuses on logistic regression, but there's also an option to use a random forest classifier (commented out).

Feel free to explore and adapt the script based on your specific requirements.

## Result - LogisticRegression:
Date = 09/07/2021
### Top 10
member_id - predicted_probability - success                                             
- 1bb7b859-68eb-4486-829c-42a0d3fba7ec-                    1.0 - False
- a9492c95-c3c0-47d9-bee6-fb8f66901854-                    1.0 - False
- 3a487e8e-1723-4709-8b36-5913973734ed-                    1.0 - False
- 10b1de10-8a8b-42b7-a8b6-a499ab13ea5f-                    1.0 - False
- 8a3c3d7d-eb8d-4f0b-bf50-eaa813881436-                    1.0 - False
- 842e832d-bf74-4624-8ab1-54a4f63d98ae-                    1.0 - False
- beaebdb5-6bf9-4535-bf63-a37447399802-                    1.0 - False
- 276a9dde-6b7e-43bf-ab5b-c70b011f5772-                    1.0 - False
- 45c458d1-639f-45fa-8b5a-82aa0db7f249-                    1.0 - True
- 59d2586a-88b0-4f5b-94af-999237b44274-                    1.0 - False

### Metric
* Number of top_10_true: 1
* Number of all_possible_true: 14
* accuracy: 0.5761674933337787
* precision: 0.03907646663539197
* recall: 0.8988043161271507
* f1: 0.07489671931956257
* roc_auc: 0.7868032370185695
* precision@k: 0.1
* recall@k: 0.07142857142857142

![img](img\importance_logistic.png)



## Result - RandomForest:
Date = 09/07/2021
### Top 10
member_id - predicted_probability - success                                             
- 37b3b319-eafd-47dc-b5c3-49badbe59a2e               1.000000    False
- aefb7c10-ee5c-407e-9d8f-7c5161358dba               1.000000    False
- ccd03555-50f2-4833-986a-5eee95b54621               1.000000    False
- 91935a3a-2c91-4036-b5d7-56cffde5f192               1.000000    False
- 34d89525-d15d-4c8a-a3cf-6dcd1bc0b6eb               0.999457    False
- 647c221a-22c7-4e5f-8aef-be7270a7b24d               0.999243    False
- 8102d986-6837-461c-b9b5-e9bbdf0433d4               0.999091    False
- afa5423e-de86-4c9d-ac26-d7e716c19ae0               0.998961    False
- 34704c74-7682-48b5-80b2-64d436efc859               0.998689    False
- 77570d0e-5e9a-4189-bc6f-436f84bf3fbe               0.998683    False

### Metric
* Number of top_10_true: 0
* Number of all_possible_true: 14
* accuracy: 0.9215473426966605
* precision: 0.08583190927450676
* recall: 0.3222513852435112
* f1: 0.13555787278415016
* roc_auc: 0.7048593402248415
* precision@k: 0.0
* recall@k: 0.0

![img](img\importance_forest.png)



