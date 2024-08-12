# Spam Classifier

## Overview
The Spam Classifier is a machine learning project designed to classify emails as either spam or ham (not spam). This project leverages Natural Language Processing (NLP) techniques to analyze email content and determine its category. 

## Project Structure
The project consists of the following key components:
- **`prepare_data.py`**: Script to process the raw data, train the model, and save the trained model and vectorizer.
- **`app.py`**: Flask application to deploy the model and provide a web interface for users to classify emails.
- **`requirements.txt`**: File listing the necessary Python libraries and dependencies.
- **`model.pkl`**: Serialized file containing the trained Logistic Regression model.
- **`vectorizer.pkl`**: Serialized file containing the TF-IDF vectorizer.
- **`index.html`**: HTML template for the web interface.

## Data Preparation
The data is sourced from `mail_data.csv`, which contains email messages and their corresponding labels (spam or ham). The `prepare_data.py` script performs the following tasks:
1. Loads the data from the CSV file.
2. Cleans the data by replacing null values and encoding labels.
3. Splits the data into training and testing sets.
4. Transforms the text data into feature vectors using TF-IDF vectorization.
5. Trains a Logistic Regression model and performs hyperparameter tuning using Grid Search.
6. Saves the trained model and vectorizer for later use.

