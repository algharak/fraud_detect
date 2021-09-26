import os

def setupdirs(dir_):
    print ("i am here")
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        return