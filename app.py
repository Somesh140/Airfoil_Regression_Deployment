import pickle
import flask
from flask import Flask,request,app,jsonify,url_for,render_template
from flask import Response
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output=model.predict(final_features)[0]

    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))

if __name__ =='__main__':
    app.run(debug=True)
