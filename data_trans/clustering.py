from log_create.logger import logging
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans



class kmeansclustering:
    def __init__(self):

        self.logobj = logging('logs\clustering_log.txt', 'ENTERED TO DATA CLUSTERING')
        self.logobj.log()

    def create_clusters(self,data):
        self.data = data
        try:
            wcss = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i)
                km.fit_predict(self.data)
                wcss.append(km.inertia_)
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('K-Means_Elbow_point.PNG')
            self.logobj.appnd_log('Elbow Method FAILED')

            # finding the value of the optimum cluster
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logobj.appnd_log('CALCULATED ELBOW POINT>>>>>>' + '\t' + str(self.kn.knee))
        except Exception as e:
            self.logobj.appnd_log('Elbow Method FAILED')
            # clustering data
        try:
            data = self.data

            X = data.iloc[:, :]
            kmean = KMeans(n_clusters= self.kn.knee)
            data['clusters'] = kmean.fit_predict(X)


            self.logobj.appnd_log('DATA CLUSTERED WITH CALCULATED ELBOW POINT>>>>>>' + '\t' + str(self.kn.knee))
        except Exception as e:
            self.logobj.appnd_log(
                'Exception while clustering. Exception message: ' + '\t' + str(
                    e))
