import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    """
    Load events and subscribers data, drop unnecessary columns, and handle missing values.

    Returns:
    events_data (pd.DataFrame): DataFrame containing events data.
    subscribers_data (pd.DataFrame): DataFrame containing subscribers data.
    """
    events_data = pd.read_csv('data/events_data.csv')
    subscribers_data = pd.read_csv('data/subscribers_data.csv')

    # Drop unnecessary columns
    subscribers_data = subscribers_data.drop(columns=['weeks_from_subscription_start'])

    # Handle missing values
    events_data.fillna(0, inplace=True)
    subscribers_data.fillna(0, inplace=True)

    # Convert date fields to datetime format
    events_data['dt'] = pd.to_datetime(events_data['dt'])
    subscribers_data['effective_date'] = pd.to_datetime(subscribers_data['effective_date'])

    # Sort data by date
    events_data.sort_values(by='dt', inplace=True)
    subscribers_data.sort_values(by='effective_date', inplace=True)

    return events_data, subscribers_data

def filter_events_by_date(events_data, specific_date, duration):
    """
    Filter events data based on a specific date and duration.

    Args:
    events_data (pd.DataFrame): DataFrame containing events data.
    specific_date (datetime): Specific date for filtering.
    duration (timedelta): Duration for filtering.

    Returns:
    filtered_df (pd.DataFrame): Filtered DataFrame.
    unique_members_in_week (numpy.ndarray): Unique member IDs in the specified week.
    """

    week_start = specific_date
    week_end = week_start + timedelta(days=6)
    members_in_week = events_data[(events_data['dt'] >= week_start) & (events_data['dt'] <= week_end)]['member_id']
    filtered_df_backward = events_data[(events_data['dt'] >= (specific_date - duration)) & (events_data['dt'] < (week_start)) & (events_data['member_id'].isin(members_in_week))]
    filtered_df_forward = events_data[(events_data['dt'] >= week_start) & (events_data['dt'] < (week_end)) & (events_data['member_id'].isin(members_in_week))]

    return filtered_df_backward, filtered_df_forward, members_in_week


def create_result_df(filtered_df, filtered_df_forward, unique_members, event_types):
    """
    Create a DataFrame with event type counts for each member (create features).

    Args:
    filtered_df (pd.DataFrame): Filtered DataFrame.
    unique_members (numpy.ndarray): Unique member IDs. (week forward)
    event_types (list): List of event types.

    Returns:
    result_df (pd.DataFrame): DataFrame with event type counts.
    """
    result_df = pd.DataFrame(index=unique_members)

    for event_type in event_types:
        event_count = filtered_df[filtered_df['event_type'] == event_type].groupby('member_id').size()
        result_df[f'{event_type}_count'] = result_df.index.map(event_count).fillna(0)

        # Count for the first, second, and third months
        if event_type == 'usage':
            for month in range(1, 4):
                month_start = filtered_df['dt'].max() - timedelta(days=(month - 1) * 30)
                month_end = filtered_df['dt'].max() - timedelta(days=month * 30)

                event_count_month = filtered_df[
                    (filtered_df['event_type'] == event_type) &
                    (filtered_df['dt'] >= month_start) &
                    (filtered_df['dt'] <= month_end)
                ].groupby('member_id').size()

                result_df[f'{event_type}_count_month_{month}'] = result_df.index.map(event_count_month).fillna(0)

            # Calculate minimum and maximum intervals between usages
            events_data_usage = filtered_df[filtered_df['event_type'] == 'usage'].copy()
            events_data_usage.sort_values(by=['member_id', 'dt'], inplace=True)
            events_data_usage['time_diff'] = events_data_usage.groupby('member_id')['dt'].diff().dt.days
            min_interval = events_data_usage.groupby('member_id')['time_diff'].min().fillna(0).astype(int)
            max_interval = events_data_usage.groupby('member_id')['time_diff'].max().fillna(0).astype(int)

            # result_df['minimum_interval'] = result_df.index.map(min_interval).fillna(0)
            # result_df['maximum_interval'] = result_df.index.map(max_interval).fillna(0)

    # Calculate if a member had a pt_sale event in the next week (1 for yes, 0 for no)
    pt_sale_count = filtered_df_forward[filtered_df_forward['event_type'] == 'pt_sale'].groupby('member_id').size()
    result_df['pt_sale'] = (result_df.index.map(pt_sale_count).fillna(0) > 0).astype(int)
    return result_df

def train_and_evaluate_model(result_df):
    """
    Train a logistic regression model and evaluate its performance.

    Args:
    result_df (pd.DataFrame): DataFrame with event type counts.

    Returns:
    model: Trained logistic regression model.
    """

    X = result_df.drop(columns=['pt_sale'])
    y = result_df['pt_sale']

    # Manually split based on member IDs
    ids = result_df.index.unique()
    split = int(0.8 * len(ids))
    train_ids, test_ids = ids[:split], ids[split:]

    # Use boolean indexing to get training and testing sets
    train_mask = result_df.index.isin(train_ids)
    test_mask = result_df.index.isin(test_ids)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    # Normalized params:
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Choose a classification algorithm (Logistic Regression)
    # model = LogisticRegression(class_weight = 'balanced')
    model = LogisticRegression()


    # Train the model
    model.fit(X_train_normalized, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test_normalized)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_normalized)[:, 1])

    return model, accuracy, precision, recall, f1, roc_auc

def train_and_evaluate_random_forest(result_df):
    """
    Train a random forest model and evaluate its performance.

    Args:
    result_df (pd.DataFrame): DataFrame with event type counts.

    Returns:
    model: Trained random forest model.
    accuracy, precision, recall, f1: Evaluation metrics.
    """
    X = result_df.drop(columns=['pt_sale'])
    y = result_df['pt_sale']

    # Manually split based on member IDs
    ids = result_df.index.unique()
    split = int(0.8 * len(ids))
    train_ids, test_ids = ids[:split], ids[split:]

    # Use boolean indexing to get training and testing sets
    train_mask = result_df.index.isin(train_ids)
    test_mask = result_df.index.isin(test_ids)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    # Choose a classification algorithm (Random Forest)
    model = RandomForestClassifier(class_weight='balanced_subsample', random_state=15)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, accuracy, precision, recall, f1, roc_auc

def make_predictions(model, result_df, subscribers_data, date_to_predict):
    """
    Make predictions for all individuals who have not taken personal training and have a subscription.

    Args:
    model: Trained logistic regression model.
    result_df (pd.DataFrame): DataFrame with event type counts.
    subscribers_data (pd.DataFrame): DataFrame containing subscribers data.
    specific_date (datetime): Specific date for predictions.

    Returns:
    subscribers_data_for_date (pd.DataFrame): Subscribers data with predicted probabilities.
    """
    prediction_df = result_df[result_df['pt_sale'] == 0].drop(columns=['pt_sale'])
    subscribers_data_for_date = subscribers_data[subscribers_data['effective_date'] == date_to_predict]

    # Filter prediction_df to include only rows with member_id present in subscribers_data_for_date
    # prediction_df = prediction_df[prediction_df.index.isin(subscribers_data_for_date['member_id'])]

    if not prediction_df.empty:
        # Continue with the predictions
        prediction_df['predicted_probability'] = model.predict_proba(prediction_df)[:, 1]

        # Rank and recommend the top 10 individuals
        top_10_recommendations = prediction_df.sort_values(by='predicted_probability', ascending=False).drop_duplicates().head(10)

        # Print the top 10 recommendations
        # print("top 10 recommendations:", top_10_recommendations[['predicted_probability']])
    else:
        print("No predictions available for the selected date.")

    # Merge the DataFrames on 'member_id' to add 'predicted_probability' to 'subscribers_data_for_date'
    subscribers_data_for_date = subscribers_data_for_date.merge(prediction_df[['predicted_probability']],
                                                                left_on='member_id', right_index=True, how='left')

    # Fill NaN values with None
    subscribers_data_for_date['predicted_probability'] = subscribers_data_for_date['predicted_probability'].where(
        subscribers_data_for_date['predicted_probability'].notna(), None)

    return subscribers_data_for_date, top_10_recommendations[['predicted_probability']]

def generate_top_10_by_segment(subscribers_data_for_date):
    """
    Generate and print top 10 recommendations for each segment.

    Args:
    subscribers_data_for_date (pd.DataFrame): Subscribers data with predicted probabilities.
    threshold (float): Threshold for considering predictions.

    Returns:
    None
    """
    sorted_dataframe = subscribers_data_for_date.sort_values(by='predicted_probability', ascending=False)
    top_10_by_segment = {}

    for segment_code in sorted_dataframe['segment_code'].unique():
        # Filter the DataFrame for the current segment_code and select the top 10 rows
        top_10_for_segment = sorted_dataframe[
            (sorted_dataframe['segment_code'] == segment_code) &
            (sorted_dataframe['predicted_probability'].notna()) &
            (sorted_dataframe['predicted_probability']> 0.1)

        ].drop_duplicates().head(10)

        # Store only the 'member_id' and 'predicted_probability' columns in the dictionary
        top_10_by_segment[segment_code] = top_10_for_segment[['member_id', 'predicted_probability']]

    # Print the top 10 members for each segment
    for segment_code, top_10_for_segment in top_10_by_segment.items():
        print(f"Top 10 for segment {segment_code}:\n{top_10_for_segment}\n")


def check_success(member_id, date_to_predict, events_data):
    """
    Check if a member has a "pt_sale" event in a week from the specific date.

    Args:
    member_id (int): Member ID to check.
    specific_date (datetime): Specific date.
    events_data (pd.DataFrame): DataFrame containing events data.

    Returns:
    success (bool): True if success, False otherwise.
    """
    end_date = date_to_predict + timedelta(days=7)  # One week from the specific date
    member_events = events_data[(events_data['member_id'] == member_id) & (events_data['event_type'] == 'pt_sale')]
    success = any((member_events['dt'] >= date_to_predict) & (member_events['dt'] <= end_date))
    return success

def collect_results(events_data, subscribers_data, start_date_for_train, date_to_predict ):

    event_types = ['app_interaction', 'personal_appointment_scheduled', 'human_communication_event',
                   'sms_sent', 'chat_message_sent', 'manual_email_sent', 'automated_email_sent',
                   'fitness_consoltation', 'usage']

    duration_in_days = 90  
    
    combine_result_dt = []

    for i in range(375): # all relevnt dates in events_data 
        date_to_collect = start_date_for_train + timedelta(days=6*i)
        filtered_df, filtered_df_forward, unique_members_in_week = filter_events_by_date(events_data, date_to_collect, timedelta(days=duration_in_days))
        result_df = create_result_df(filtered_df, filtered_df_forward, unique_members_in_week, event_types)
        combine_result_dt.append(result_df) 

    combine_result_dt = pd.concat(combine_result_dt)

    model, accuracy, precision, recall, f1, roc_auc = train_and_evaluate_model(combine_result_dt)
    # model, accuracy, precision, recall, f1, roc_auc = train_and_evaluate_random_forest(combine_result_dt)
    subscribers_data_for_date, top_10_predictions = make_predictions(model, combine_result_dt, subscribers_data, date_to_predict)

    generate_top_10_by_segment(subscribers_data_for_date)

    top_10_predictions['success'] = False
    # Check success for each member in top 10 predictions
    for idx in top_10_predictions.index:
        top_10_predictions.loc[idx, 'success'] = check_success(idx, date_to_predict, events_data)

    top_10_true =  top_10_predictions['success'].sum()
    
    # Check all members that buy pt_sale:
    members_in_week = events_data[(events_data['dt'] >= date_to_predict) & (events_data['dt'] <= (date_to_predict+timedelta(days=6)))]['member_id']
    evente_week_forward = events_data[(events_data['dt'] >= date_to_predict) & (events_data['dt'] < (date_to_predict+timedelta(days=6))) & (events_data['member_id'].isin(members_in_week))]
    # Calculate if a member had a pt_sale event in the next week (1 for yes, 0 for no)
    all_possible_true = evente_week_forward[evente_week_forward['event_type'] == 'pt_sale'].groupby('member_id').size()
    all_possible_true = (all_possible_true.index.map(all_possible_true).fillna(0) > 0).astype(int).sum()
    
    # Store relevant information for comparison
    results = {
    'model': model,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'roc_auc': roc_auc,
    'precision@k': top_10_true/10 ,
    'recall@k': top_10_true/all_possible_true,
    }

    return results, model


if __name__ == "__main__":
    start_date_for_train = datetime(2015, 9, 11) # start of relevant data to train
    date_to_predict = datetime(2021, 7, 9) 
    events_data, subscribers_data = load_and_preprocess_data()
    results, model = collect_results(events_data, subscribers_data, start_date_for_train, date_to_predict)

    for key, value in results.items():
        print("------------------------")
        print(f"{key}: {value}")
