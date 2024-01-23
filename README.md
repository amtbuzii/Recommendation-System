#Vi labs

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
