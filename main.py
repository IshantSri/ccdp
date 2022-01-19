from flask import Flask,render_template,request,redirect
from prediction_pre_process import PREDICTION
from data_trans import data_transform

from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

@app.route("/",methods = ['GET'])
def home():
    return render_template('index.html')

@app.route("/", methods = ["POST"])
def choose_file():
    data= request.files['data']
    dp = './dp' + data.filename
    data.save(dp)
    return render_template('index.html')







if __name__ == "__main__":
    app.run(debug=True)


