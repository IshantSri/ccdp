from log_create.logger import logging
from data_trans.data_transform import data_transform
from data_trans.clustering import kmeansclustering

file = data_transform('raw_data.csv')
file.missing_value()
file.chng_vrbl_name()
file.data_type()
file.mapping()
file.remove_id()
file.csv()
k = kmeansclustering()
k.create_clusters()

