from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, render_template
import pickle
import pandas as pd 

app = Flask(__name__)

model_file = open('nb.pkl', 'rb')
model = pickle.load(open('nb.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', insurance_cost=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    
    df= pd.read_csv("data_gender.csv")
    X = df['nama']
    
    cv = CountVectorizer()
    X = cv.fit_transform(X) 
    
    name = [x for x in request.form.values()]

    data = cv.transform([str(name)])   
    prediction = model.predict(data)

    if prediction == 'm':
        output = 'Laki-Laki'
    else:
        output = 'Perempuan'

    return render_template('index.html', sex=output, names=name)


if __name__ == '__main__':
    app.run(debug=True)