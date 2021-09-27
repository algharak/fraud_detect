# -*- coding: utf-8 -*-
"""Data Loader"""

#import tensorflow_datasets as tfds
import kaggle
import os
import utils.setup_dirs
import utils.kaggle_dataset

setupdirs = utils.setup_dirs.setupdirs
kaggledset = utils.kaggle_dataset.kaggle_dset


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        lpath = data_config.dataset.path
        setupdirs(lpath)
        dset = kaggledset(data_config)
        return dset
