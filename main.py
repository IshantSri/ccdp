from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model= pickle.load(open('ccdp.pkl','rb'))
d= datagetter('data.csv')
da= d.getdata()

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    prediction=model.predict()
    print(prediction)

    return str(np.round(prediction,2))



if __name__=='__main__':
    app.run()