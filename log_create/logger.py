import datetime
class logging:
    def __init__(self,log_file,log_msg):
        self.log_msg = log_msg
        self.log_file = log_file
        self.now = datetime.datetime.now()

    def log(self):
        file = open(self.log_file, 'w')
        file.write(str(self.now) + " >>>>>>>>>" + "\t" + str(self.log_msg) + "\n")

    def appnd_log(self,comment):
        self.comment = comment
        file  = open(self.log_file,'a+')
        file.write(str(self.now) + " >>>>>>>>>" + "\t" + str(self.comment) + "\n")