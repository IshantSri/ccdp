from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay
from log_create.logger import logging
import matplotlib.pyplot as plt

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By:
                Version:
                Revisions:

                """

    def __init__(self,train_x,train_y,test_x,test_y):
        self.train_x=train_x
        self.train_y=train_y
        self.test_x = test_x
        self.test_y = test_y
        self.logger_object = logging('log\model_select_log.txt', 'ENTERED TO DATA MODEL SELECTION MODULE')
        self.logger_object.log()
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_random_forest(self):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: model with the best parameters


                                Written By:
                                Version:
                                Revisions:

                        """
        self.logger_object.appnd_log("FINDING BEST PARAMETER FOR RANDOM FOREST")
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(self.train_x, self.train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(self.train_x, self.train_y)
            self.logger_object.appnd_log(
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.appnd_log(
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.appnd_log(
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters


                                        Written By:
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.appnd_log(
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.02, 0.1, 0.01, 0.05,0.001,1],
                'max_depth': [3],
                'n_estimators': [10, 1000,2000,250]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(self.train_x, self.train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(self.train_x, self.train_y)
            self.logger_object.appnd_log(
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.appnd_log(
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.appnd_log(
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By:
                                                Version:
                                                Revisions:

                                        """
        self.logger_object.appnd_log(
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost()
            self.prediction_xgboost = self.xgboost.predict(self.test_x) # Predictions using the XGBoost Model

            if len(self.test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(self.test_y, self.prediction_xgboost)
                self.recall = recall_score(self.test_y, self.prediction_xgboost)
                self.pricision = precision_score(self.test_y, self.prediction_xgboost)
                self.matrix = confusion_matrix(self.test_y,self.prediction_xgboost)
                cm = confusion_matrix(self.test_y, self.prediction_xgboost)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig('cm.PNG')
                self.logger_object.appnd_log( 'Accuracy for XGBoost:' + str(self.xgboost_score)+
                                              'with recall '+str(self.recall)+'and precision '+str(self.pricision)+
                                              " where confusion mtrx is "+str(self.matrix))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(self.test_y, self.prediction_xgboost) # AUC for XGBoost
                self.pricision = precision_score(self.test_y,self.prediction_xgboost)
                self.recall = recall_score(self.test_y,self.prediction_xgboost)
                self.matrix = confusion_matrix(self.test_y, self.prediction_xgboost)
                self.accuracy = accuracy_score(self.test_y,self.prediction_xgboost)

                self.logger_object.appnd_log( 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC
                self.logger_object.appnd_log('Accuracy for XGBoost:' + str(self.accuracy) +
                                             'with recall '+str(self.recall)+'and precision '+str(self.pricision)+
                                             " where confusion mtrx is "+str(self.matrix))



            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest()
            self.prediction_random_forest=self.random_forest.predict(self.test_x) # prediction using the Random Forest Algorithm

            if len(self.test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(self.test_y,self.prediction_random_forest)
                self.logger_object.appnd_log( 'Accuracy for RF:' + str(self.random_forest_score))
                self.matrix2 = confusion_matrix(self.test_y, self.prediction_random_forest)
                self.pricision2 = precision_score(self.test_y, self.prediction_random_forest)
                self.recall2 = recall_score(self.test_y, self.prediction_random_forest)
                self.logger_object.appnd_log(
                    'AcC for:' + str(self.random_forest_score) +
                    'xgboost with recall ' + str(self.recall2) + 'and precision ' + str(
                        self.pricision2) + " where confusion mtrx is " + str(self.matrix2))
            else:
                self.random_forest_score = roc_auc_score(self.test_y, self.prediction_random_forest) # AUC for Random Forest

                self.matrix2 = confusion_matrix(self.test_y, self.prediction_random_forest)
                self.pricision2 = precision_score(self.test_y, self.prediction_random_forest)
                self.recall2= recall_score(self.test_y, self.prediction_random_forest)
                self.logger_object.appnd_log(
                    'AUC :' + str(self.random_forest_score) +
                    'xgboost with recall ' + str(self.recall2) + 'and precision ' + str(
                        self.pricision2) + " where confusion mtrx is " + str(self.matrix2))



            #comparing the two models
            if(self.recall2<  self.recall):
                cm = confusion_matrix(self.test_y, self.prediction_xgboost)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig('cm.PNG')
                self.logger_object.appnd_log( 'AUC for XGBoost:' + str(self.xgboost_score)+'Accuracy for XGBoost:' + str(self.accuracy)+
                    'xgboost with recall ' + str(self.recall) + 'and precision ' + str(
                        self.pricision) + " where confusion mtrx is " + str(self.matrix))
                return 'XGBoost',self.xgboost

            else:
                cm = confusion_matrix(self.test_y, self.prediction_random_forest)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig('RF.PNG')
                self.logger_object.appnd_log(
                    'AUC for XGBoost:' + str(self.random_forest_score) +
                    'xgboost with recall ' + str(self.recall2) + 'and precision ' + str(
                        self.pricision2) + " where confusion mtrx is " + str(self.matrix2))
                self.logger_object.appnd_log('AUC for XGBoost:' + str(self.random_forest_score))  # Log AUC

                self.logger_object.appnd_log(
                                             'RF with recall ' + str(self.recall2) + 'and precision ' + str(self.pricision2) +" where confusion mtrx is " + str(self.matrix2))
                return 'RandomForest',self.random_forest


        except Exception as e:
            self.logger_object.appnd_log(
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.appnd_log(
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

