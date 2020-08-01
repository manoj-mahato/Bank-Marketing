import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('bank_marketing_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,10)
    prediction = model.predict(final_features)

    output = prediction[0]
    if output == 0:
        pt = 'No, customer will Not subscribe to the term deposit.'
    elif output == 1:
        pt = 'Yes, customer will subscribe to the term deposit.'
    else:
        pt = 'error'

    return render_template('index.html', prediction_text= pt)


if __name__ == "__main__":
    app.run(debug=True)