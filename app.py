# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:14:27 2020

@author: nlktoen
"""

from flask import Flask, request, url_for, redirect, render_template, jsonify
import flask
import pickle
import numpy as np
import pandas as pd

app = flask.Flask(__name__, template_folder='C:/Kat/Tuyen/Project')
model = pickle.load(open("model.pkl","rb"))
scaler= pickle.load(open("Scaler","rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['Post'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final])
    data_unseen_scaled = scaler.transform(data_unseen)
    prediction = model.predict(data_unseen_scaled)
    if prediction == 0:
        text="M"
    elif prediction == 1:
        text="B"
    cl=max(max((model.predict_proba(data_unseen_scaled))))*100
    return render_template('index.html',prediction_text='Predicted class will be: {}. Confidence level: {}%'.format(text,cl))

if __name__ == "__main__":
    app.run(debug=True)
