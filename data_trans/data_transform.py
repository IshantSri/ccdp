from log_create.logger import logging
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



class data_transform:

    def __init__(self):
        self.log_file = 'logs/data_transform.txt'
        self.log_obj = logging(self.log_file, 'ENTERED TO DATA TRANSFORMATION')
        self.log_obj.log()

    def null_value(self,data):
        self.data = data
        self.check = self.data.isnull()
        self.log_obj.appnd_log('FINDING NULL VALUES')
        try:
            for i in self.check:
                if i == True:
                    imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
                    self.new_array = imputer.fit_transform(self.data)  # impute the missing values
                    # convert the nd-array returned in the step above to a Dataframe
                    self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)
                    self.log_obj.appnd_log('Imputing missing values Successful.')
                else:
                    self.log_obj.appnd_log('NO NULL VALUES FOUND')
        except Exception as e:
            self.log_obj.appnd_log('SOMETHING WENT WRONG AT DATA TRANSFORMATION FAILED TO TREAT MISSING VALUES-> ' + str(e))


    def chng_vrbl_name(self,data):
        self.log_obj.appnd_log('CHANGING_VRBL_NAME')
        self.data = data
        try:
            self.data = self.data.rename(columns={'default.payment.next.month':'y'})
            self.data = self.data.rename(columns={'PAY_0': 'PAY_1'})

            self.log_obj.appnd_log('VRBL_NAME CHANGED')
        except Exception as e:
            self.log_obj.appnd_log('FAILED TO CHNG_VRBL_NAME ->'+str(e))


    def data_type(self,data):
        self.data = data
        self.log_obj.appnd_log('ENTERED TO DATA_TYPE')
        try:

            self.data['SEX'] = self.data['SEX'].astype('category')
            self.data['EDUCATION'] = self.data['EDUCATION'].astype('category')
            self.data['PAY_1'] = self.data['PAY_1'].astype('category')
            self.data['PAY_2'] = self.data['PAY_2'].astype('category')
            self.data['PAY_3'] = self.data['PAY_3'].astype('category')
            self.data['PAY_4'] = self.data['PAY_4'].astype('category')
            self.data['PAY_5'] = self.data['PAY_5'].astype('category')
            self.data['PAY_6'] = self.data['PAY_6'].astype('category')
            self.data['y'] = self.data['y'].astype('category')
            self.log_obj.appnd_log('CHANGED DATA TYPE')
        except Exception as e:
            self.log_obj.appnd_log('FAILED TO CHANGE DATA TYPE->' + str(e))


    def mapping(self,data):
        self.log_obj.appnd_log('ENTERED TO MAPPING')
        self.data = data

        try:
            self.data["EDUCATION"] = self.data["EDUCATION"].map({0: 4, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4})
            self.data['MARRIAGE'] = self.data['MARRIAGE'].map({0: 3, 1: 1, 2: 2, 3: 3})
            self.data['PAY_1'] = self.data['PAY_1'].map({-1: 0, -2: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
            self.data['PAY_2'] = self.data['PAY_2'].map({-1: 0, -2: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
            self.data['PAY_3'] = self.data['PAY_3'].map({-1: 0, -2: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
            self.data['PAY_4'] = self.data['PAY_4'].map({-1: 0, -2: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
            self.data['PAY_5'] = self.data['PAY_5'].map({-1: 0, -2: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
            self.data['PAY_6'] = self.data['PAY_6'].map({-1: 0, -2: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
            self.log_obj.appnd_log('MAPPING COMPLETED')
        except Exception as e:
            self.log_obj.appnd_log('FAILED TO MAP DATA->' + str(e))



    def zero_stdv(self,data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.log_obj.appnd_log(
                               'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.data = data
        self.columns = self.data.columns
        self.data_n = self.data.describe()
        self.col_to_drop = []
        try:
            for i in self.columns:
                if (self.data_n[i]['std'] == 0):  # check if standard deviation is zero
                    self.col_to_drop.append(i)  # prepare the list of columns with standard deviation zero

                    self.log_obj.appnd_log(
                                   'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            self.useful_data = self.data.drop(labels=self.col_to_drop, axis=1)
        except Exception as e:
            self.log_obj.appnd_log(
                                   'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(
                                       e))

    def remove_id(self,data):
        self.log_obj.appnd_log('entered to remove_id')
        self.data = data
        try:
            self.no_id_data = self.data.drop("ID",axis=1)
            self.log_obj.appnd_log('id column removed')
            return self.no_id_data
        except Exception as e:
            self.log_obj.appnd_log('failed to remove id column >>>>>>>' + str(e))



    def std_scaling(self,data):
        self.data = data
        self.log_obj.appnd_log('entered to standard scaling')
        try:
            self.scaling = StandardScaler()
            self.x =self.data.drop(['y'],axis=1)
            self.X =self.scaling.fit_transform(self.x)
            self.log_obj.appnd_log('standard scaling completed')
            return self.X
        except Exception as e:
            self.log_obj.appnd_log('standard scaling failed >>>>>>' + str(e))



    def label(self,data):
        self.data = data
        self.log_obj.appnd_log('entered to dependent label extraction')
        try:
            self.y = self.data['y']
            self.log_obj.appnd_log('dependent label extracted')
            return self.y
        except Exception as e:
            self.log_obj.appnd_log('dependent label extraction failed >>>>>>' + str(e))




    def tocsv(self,data):
        self.data = data
        self.log_obj.appnd_log('saving data')
        try:
           self.data.to_csv('new_transformed.csv',index=False)
           self.log_obj.appnd_log('data saved')
        except Exception as e:
            self.log_obj.appnd_log('failed to save data >>>>>' + str(e))