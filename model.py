import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load the football data
matches = pd.read_csv("football_data.csv")

# Convert 'date' to datetime
matches['date'] = pd.to_datetime(matches['date'])

# Feature Engineering
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(":.+", "", regex=True).astype('int')
matches['day_code'] = matches['date'].dt.dayofweek
matches['target'] = (matches['result'] == 'W').astype('int')

# Initialize RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Split the data into training and test sets based on a specific date
split_date = '2024-01-01'
train = matches[matches['date'] < split_date]
test = matches[matches['date'] >= split_date]

# Predictors used in the model
predictors = ['venue_code', 'opp_code', 'hour', 'day_code']

# Fit the model on training data and predict on test data
rf.fit(train[predictors], train['target'])
preds = rf.predict(test[predictors])

# Evaluate accuracy and precision
acc = accuracy_score(test['target'], preds)
prec = precision_score(test['target'], preds)

# Print results
print("Accuracy:", acc)
print("Precision:", prec)

# Function to compute rolling averages for selected columns
def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Selected columns for rolling averages
cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages to each team
matches_rolling = matches.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Print to verify the columns after applying rolling averages
print("Columns in matches_rolling:", matches_rolling.columns)

# Function to make predictions and calculate precision
def make_predictions(data, predictors):
    split_date = '2024-01-01'
    train = data[data['date'] < split_date]
    test = data[data['date'] >= split_date]

    # Fit the model on training data and predict on test data
    rf.fit(train[predictors], train['target'])
    preds = rf.predict(test[predictors])

    # Create DataFrame with actual and predicted values
    combined = pd.DataFrame(dict(actual=test['target'], predicted=preds), index=test.index)

    # Calculate precision score
    precision = precision_score(test['target'], preds)
    return combined, precision

# Make predictions using matches_rolling and the updated predictors
combined, precision = make_predictions(matches_rolling, predictors + new_cols)

# Merge the prediction results with additional match information
if combined is not None:
    combined = combined.merge(matches_rolling[['date', 'team', 'opponent', 'result']], left_index=True, right_index=True)
    print(combined.head())
    print("Precision:", precision)

# Class for custom mapping of team names
class MissingDict(dict):
    __missing__ = lambda self, key: key

# Custom mapping for team names
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

# Apply mapping to the team names
combined['new_team'] = combined['team'].map(mapping)

# Merge data to analyze the results of the model's predictions
merged = combined.merge(combined, left_on=['date', 'new_team'], right_on=['date', 'opponent'])

# Analyze cases where the model predicted team A would win and team B would lose
result_analysis = merged[(merged['predicted_x'] == 1) & (merged['predicted_y'] == 0)]['actual_x'].value_counts()

# Print the result analysis
print(result_analysis)
