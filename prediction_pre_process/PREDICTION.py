from log_create.logger import logging
from data_trans.data_transform import data_transform
from data_trans.clustering import kmeansclustering



class predict:
    def __init__(self):
        self.log_file = './log/Prediction.txt'
        self.logobj = logging(self.log_file, 'ENTERED TO PREDICTION STAGE')
        self.logobj.log()
        self.cluster = kmeansclustering()
    def pred_val(self,data):
        self.data = data
        self.trans = data_transform()

        try:
            self.logobj.appnd_log('PREPROCESSING THE DATA')
            drop_id = self.trans.remove_id(self.data)
            self.logobj.appnd_log('ID REMOVED')
            changed_var = self.trans.chng_vrbl_name(drop_id)
            self.logobj.appnd_log('VARIABLE NAME CHANGED')
            dtype_data = self.trans.data_type(changed_var)
            self.logobj.appnd_log('DATA TYPE CHANGED')
            map = self.trans.mapping(dtype_data)
            self.logobj.appnd_log('MAPPING COMPLETED')
            impu = self.trans.value_imputer(map)
            self.logobj.appnd_log('TREATED NAN VALUE')
            scaled_data = self.trans.std_scaling(impu)
            self.logobj.appnd_log('STANDARD SCALING COMPLETED')
            self.final_data = self.trans.zero_stdv(scaled_data)
            self.logobj.appnd_log('TREATED 0 STANDARD_DEVIATION COLUMNS ')
            self.logobj.appnd_log('PREPROCESSING COMPLETED')
            return  self.final_data
        except Exception as e:
            self.logobj.appnd_log('PREPROCESSING FAILED >>>>>>' + str(e))




