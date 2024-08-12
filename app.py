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
