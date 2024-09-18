# RandomForest-Football-Analysis
This project uses a RandomForestClassifier to predict the outcome of football matches based on historical data and statistics such as venue, opponent, time, and various match statistics. The goal is to classify whether a team will win or lose a match.

# Dataset
The dataset includes various features such as:

Date of the match
Venue
Opponent
Match time
Goals for (gf), goals against (ga)
Shots (sh), shots on target (sot)
Distance covered (dist)
Free kicks (fk), penalties (pk), penalty attempts (pkatt)
The target variable is the match result (Win/Not Win).

# Features
The main features used in the model include:

Venue code: Encoded value for the venue.
Opponent code: Encoded value for the opposing team.
Hour: Match start time (converted from string to integer).
Day of the week: Encoded value for the day of the match.
Rolling Averages: Averages of key statistics like goals, shots, and distances over the previous three matches.

# Model
The RandomForestClassifier is used with the following hyperparameters:

n_estimators=50: Number of decision trees.
min_samples_split=10: Minimum number of samples to split a node.
random_state=1: To ensure reproducibility.
