import os
import kaggle
import pandas as pd



def kaggle_dset(cfg):
    """ import the kaggle credit card fraud detection data set and return it in padas dataframe """
    """ inputs:
            configuration json
        outputs:
            a dataframe of the entire dataset"""

    os.environ['KAGGLE_USERNAME'] = cfg.dataset.kaggle_username
    os.environ['KAGGLE_KEY'] = cfg.dataset.kaggle_key
    target_file = os.path.join(cfg.dataset.path, cfg.dataset.lfile)
    if not os.path.isfile(target_file):
        print ("it does not exist")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(cfg.dataset.name, path=cfg.dataset.path,unzip=True)
    return pd.read_csv(target_file)


'''
 
    os.environ['KAGGLE_USERNAME'] = cfg.dataset.kaggle_username
    os.environ['KAGGLE_KEY'] = cfg.dataset.kaggle_key
    if not os.path.exists(dir_):
        os.makedirs(dir_)
   


kaggle.api.authenticate()

kaggle.api.dataset_download_files('The_name_of_the_dataset', path='the_path_you_want_to_download_the_files_to', unzip=True)   

'''