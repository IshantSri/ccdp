from log_create.logger import logging
from data_trans.clustering import kmeansclustering
from data_trans.data_transform import data_transform
from data_loader.importing_raw_data import datagetter
from model_select.best_algo import Model_Finder

class train:
    def __init__(self,data_path):
        self.logobj = logging('log\model_building_log.txt', 'ENTERED TO DATA MODEL BUILDING MODULE')
        self.logobj.log()
        self.data_path = datagetter(data_path)
        self.data = self.data_path.getdata()
        self.trans = data_transform()
        self.cluster = kmeansclustering()


    def training(self):
        self.logobj.appnd_log('TRAINING STARTED')
        try:
            self.logobj.appnd_log('PREPROCESSING THE DATA')
            self.drop_id =self.trans.remove_id(self.data)
            self.logobj.appnd_log('ID REMOVED')
            self.changed_var = self.trans.chng_vrbl_name(self.drop_id)
            self.logobj.appnd_log('VARIABLE NAME CHANGED')
            self.dtype_data = self.trans.data_type(self.changed_var)
            self.logobj.appnd_log('DATA TYPE CHANGED')
            self.map = self.trans.mapping(self.dtype_data)
            self.logobj.appnd_log('MAPPING COMPLETED')
            self.null = self.trans.null_value(self.map)
            self.logobj.appnd_log('TREATED NULL VALUE')
            self.scaled_data = self.trans.std_scaling(self.null)
            self.logobj.appnd_log('STANDARD SCALING COMPLETED')
            self.final_data = self.trans.zero_stdv(self.scaled_data)
            self.logobj.appnd_log('TREATED 0 STANDARD_DEVIATION COLUMNS ')
            self.logobj.appnd_log('PREPROCESSING COMPLETED')
        except Exception as e:
            self.logobj.appnd_log('PREPROCESSING FAILED >>>>>>'+ str(e))

        try:
            self.logobj.appnd_log('CLUSTERING DATA')
            self.clustered_data = self.cluster.create_clusters(self.final_data)
            self.logobj.appnd_log('DATA CLUSTERED')
        except Exception as e:
            self.logobj.appnd_log('CLUSTERING FAILED>>>>>' + str(e))

        try:
            self.logobj.appnd_log('LABEL EXTRACTION STARTED')
            self.y = self.trans.label(self.clustered_data)
            self.x =self.clustered_data.drop(['y'],axis=1)
            self.logobj.appnd_log('LABEL EXTRACTION COMPLETED')
        except Exception as e:
            self.logobj.appnd_log('LABEL EXTRACTION FAILED>>>>>>' + str(e))

        try:










