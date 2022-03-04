import pandas as pd
from prediction_pre_process.PREDICTION import predict
from data_loader.importing_raw_data import datagetter
import pickle
import numpy as np

from data_trans.data_transform import data_transform

"""training the model"""
#from model_operation.model_building import train

#a = train('data.csv')
#a.training()


""" prediction on data set"""
p = predict()

m = pickle.load(open('ccdp.pkl', 'rb'))

d = datagetter('data.csv')# data set on which prediction is to done
da = d.getdata()

dt = data_transform()
d = p.pred_val(da)
d = dt.label(d)
print(d)
prediction = m.predict_proba(d)
# print(pd.DataFrame(prediction,columns=['repayment','repayment']))
prediction = pd.DataFrame(prediction, columns=['repayment', 'default'])

da['prob_default'] = np.round(prediction['default'], 2)
da['prob_repayment'] = np.round(prediction['repayment'], 2)
dt.tocsv(da, "df.csv")

