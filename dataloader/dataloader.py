# -*- coding: utf-8 -*-
"""Data Loader"""

#import tensorflow_datasets as tfds
import kaggle
import os
import utils.setup_dirs
setupdirs = utils.setup_dirs.setupdirs
class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        lpath = data_config.dataset.path
        lfile = data_config.dataset.lfile
        kfile = data_config.dataset.kfile
        kurl = data_config.dataset.kurl
        setupdirs(lpath)
        os.environ['KAGGLE_USERNAME'] = data_config.dataset.kaggle_username
        os.environ['KAGGLE_KEY'] = data_config.dataset.kaggle_key
        return "this","that"


'''

CFG = {
    "dataset": {
        "path": "./dataset",
        "lfile": "creditcard.csv"
    },
    "kaggle":{
        "username":"gharakhanian",
        "key":"abe9ee2ccc9321fcaaa6c4d71306d92d",
        "kfile":"creditcard.csv",
        "kurl":"https://www.kaggle.com/mlg-ulb/creditcardfraud"
    }
}

'''