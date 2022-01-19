import pandas as pd
from prediction_pre_process.PREDICTION import predict
from data_trans import data_transform as dt
from data_loader.importing_raw_data import datagetter
import pickle
import numpy as np




predict = predict()
datatrans = dt.data_transform()
model = pickle.load(open('ccdp.pkl','rb'))






m = pickle.load(open('ccdp.pkl', 'rb'))

d = datagetter('data.csv')
da = d.getdata()

d = predict.pred_val(da)
d = datatrans.label(d)
print(d)
prediction = model.predict_proba(d)
# print(pd.DataFrame(prediction,columns=['repayment','repayment']))
prediction = pd.DataFrame(prediction, columns=['repayment', 'default'])

da['prob_default'] = np.round(prediction['default'], 2)
da['prob_repayment'] = np.round(prediction['repayment'], 2)
datatrans.tocsv(da, "exp.csv")
a = pd.read_csv("exp.csv")
print(a)


