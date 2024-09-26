Certainly! Here’s an updated `README.md` with detailed explanations for each code component:

---

# Spam Classifier

## Overview
The Spam Classifier project is a machine learning application designed to classify emails as either spam or ham (non-spam). The project utilizes a Logistic Regression model combined with TF-IDF feature extraction and hyperparameter tuning to achieve effective spam detection. A Flask web application is used to interact with the model.

## Project Structure
The project contains the following files:
- `prepare_data.py`: Script for data preprocessing, model training, and saving.
- `app.py`: Flask application that serves the model and handles user input.
- `model.pkl`: Saved Logistic Regression model.
- `vectorizer.pkl`: Saved TF-IDF vectorizer.

## Files and Code Explanation

### `prepare_data.py`

```python
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
```

**Explanation:**
- **Data Loading:** Reads email data from a CSV file and handles any file-not-found errors.
- **Data Preprocessing:** Replaces null values with empty strings and converts categorical labels ('spam', 'ham') into numerical format (0, 1).
- **Feature Extraction:** Uses TF-IDF with bi-grams to convert email text into numerical features.
- **Model Training:** Splits the data into training and test sets, performs hyperparameter tuning using Grid Search, and trains the best Logistic Regression model.
- **Saving Artifacts:** Saves the trained model and vectorizer using `pickle` for later use.

### `app.py`

```python
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        input_text = request.form['message']
        input_data_features = vectorizer.transform([input_text])
        prediction = model.predict(input_data_features)
        result = 'Ham mail' if prediction[0] == 1 else 'Spam mail'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
```

**Explanation:**
- **Flask Setup:** Initializes a Flask application.
- **Loading Artifacts:** Loads the saved model and vectorizer from `pickle` files.
- **Index Route:** Handles both GET and POST requests. On a POST request, it processes the input text, classifies it using the model, and returns the classification result.
- **Run Application:** Starts the Flask server in debug mode.

### `templates/index.html`

Here’s a simple HTML template for the user interface:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Spam Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #000000;
            color: #ffffff;
            margin: 0;
            padding: 0;
            text-align: center;
        }
        h1 {
            color: #ffffff;
            margin-top: 50px;
        }
        form {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 8px;
            display: inline-block;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            margin-top: 20px;
            width: 70%; /* Adjusted width for the form */
            max-width: 800px; /* Maximum width to prevent it from getting too wide */
            text-align: left; /* Align text inside the form to the left */
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 4px;
            font-size: 16px;
            background-color: #222;
            color: #eee;
            box-sizing: border-box; /* Ensure padding is included in width */
        }
        input[type="submit"] {
            background-color: #4caf50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        h2 {
            margin-top: 20px;
        }
        p {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Spam Classifier</h1>
    <form method="post">
        <textarea name="message" placeholder="Enter your message here..."></textarea><br>
        <input type="submit" value="Classify">
    </form>
    {% if result %}
    <h2>Result:</h2>
    <p>{{ result }}</p>
    {% endif %}
</body>
</html>
```

## Getting Started

### Prerequisites
- Python 3.x
- Flask
- scikit-learn
- pandas

### Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/Pkanagalakshmi/Spam-Classifier.git
   cd Spam-Classifier
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Preparing Data
Run the `prepare_data.py` script to preprocess the data, train the model, and save the model and vectorizer:

```bash
python prepare_data.py
```

### Running the Application
1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Open a web browser and navigate to `http://127.0.0.1:5000/` to access the application.

### Usage
- On the main page, enter the email text in the textarea and click "Classify."
- The application will display whether the email is classified as "Ham mail" or "Spam mail."

