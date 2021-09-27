# -*- coding: utf-8 -*-
"""Model config in json format"""


'''
CFG = {
    "data": {
        "path": "oxford_iiit_pet:3.*.*",
        "image_size": 128,
        "load_with_info": True
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epoches": 20,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}



'''
CFG = {
    "dataset": {
        "path": "./dataset",
        "lfile": "creditcard.csv",
        "kaggle_username": "gharakhanian",
        "kaggle_key": "abe9ee2ccc9321fcaaa6c4d71306d92d",
        "kfile":"creditcard.csv",
        "name":"mlg-ulb/creditcardfraud"
    },
    "model": {
        "input": "model params to be added later"
        },
    "train": {
        "input": "train params to be added later"
        },
    "data": {
        "input": "data params to be added later"
        }

}
