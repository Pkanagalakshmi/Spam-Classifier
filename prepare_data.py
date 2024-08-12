import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the data from CSV file
file_path = r'C:\Users\Kanaga\Downloads\mail_data.csv'
try:
    raw_mail_data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the file path.")
    exit()

# Replace null values with an empty string
mail_data = raw_mail_data.fillna('')

# Convert 'spam' to 0 and 'ham' to 1
mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

# Separate the data into features and labels
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction with bi-grams
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2))
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert labels to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Hyperparameter tuning with Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_features, Y_train)

# Train the best model
model = grid_search.best_estimator_

# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(feature_extraction, vectorizer_file)

print("Model and vectorizer saved successfully!")
