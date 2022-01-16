from log_create.logger import logging
from data_trans.clustering import kmeansclustering
from data_trans.data_transform import data_transform
from data_loader.importing_raw_data import datagetter
from model_select.best_algo import Model_Finder
from sklearn.model_selection import train_test_split
import pickle


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
            drop_id =self.trans.remove_id(self.data)
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
        except Exception as e:
            self.logobj.appnd_log('PREPROCESSING FAILED >>>>>>'+ str(e))


        try:
            self.logobj.appnd_log('CLUSTERING DATA')
            self.clustered_data = self.cluster.create_clusters(self.final_data)
        except Exception as e:
            self.logobj.appnd_log('CLUSTERING FAILED>>>>>' + str(e))


        try:
           # getting the unique clusters from our dataset
            list_of_clusters = self.clustered_data['clusters'].unique()
            for i in list_of_clusters:
                self.cluster_data = self.clustered_data[self.clustered_data['clusters']==i] # filter the data for one cluster
                self.logobj.appnd_log('WORKING ON CLUSTER>>>> '+str(i))
                #feature extraction
                self.cluster_features= self.cluster_data.drop(['y','clusters'], axis=1)

                self.cluster_label= self.cluster_data['y']
                self.logobj.appnd_log('EXTRACTED LABEL AND FEATURE COULMN')
                # splitting the data into training and test set for each cluster one by one
                self.x_train,self.y_train,self.x_test, self.y_test = train_test_split(self.cluster_features, self.cluster_label, test_size=0.3, random_state=300)
                self.logobj.appnd_log('TRAIN TEST SPLIT COMPLETED')

                self.model_finder=Model_Finder(self.x_train, self.x_test, self.y_train, self.y_test) # object initialization


                self.logobj.appnd_log('ENTERED TO BEST MODEL FINDING STAGE')


                #getting the best model for each of the clusters
                self.best_model_name,self.best_model=self.model_finder.get_best_model()
                self.logobj.appnd_log('MODEL FINDING COMPLETED BEST MODEL IS'
                                      + str(self.best_model_name)+" WITH "+str(self.best_model))

        except Exception as e:
            self.logobj.appnd_log('FAILED TO FIND THE BEST MODEL>>> '+str(e))


                # saving the best model to the directory.
        try:
            pickle.dump(self.best_model,open('ccdp.pkl','wb'))
            self.logobj.appnd_log("SAVED MODEL AS ccdp.pkl")
        except Exception as e:
            self.logobj.appnd_log("FAILED TO SAVE THE MODEL>>>>>> " + str(e))
















