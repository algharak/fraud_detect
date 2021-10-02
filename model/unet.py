# -*- coding: utf-8 -*-
"""Unet model"""
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score, roc_curve, precision_score
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import cross_val_score
#from imblearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader

# external
import tensorflow as tf

class UNet(BaseModel):
    """Unet Model Class"""
    def __init__(self, config):
        super().__init__(config)
    def load_data(self):
        """Loads and stores data in the proper path/file """
        self.dataset = DataLoader().load_data(self.config)
        self.batch_size = self.config.train.batch_size
        self.epoches = self.config.train.epoches
        self.val_subsplits = self.config.train.val_subsplits
        self._preprocess_data()
    def _preprocess_data(self):
        self.train_dataset = self.dataset.sample(frac=self.config.train.val_subsplits, random_state=0)
        self.test_dataset = self.dataset.drop(self.train_dataset.index)
        all_cols = self.train_dataset.columns
        label_col   = [self.config.train.label_colname]
        feat_cols   = [i for i in all_cols if i not in label_col]
        self.y_te, self.y_tr = self.test_dataset[label_col],self.train_dataset[label_col]
        self.X_te, self.X_tr = self.test_dataset[feat_cols],self.train_dataset[feat_cols]
        self.input_shape = len(feat_cols)

    def build(self):
        """ Builds sklearn model """
        self.model = LogisticRegression(penalty=self.config.model.penalty,
                                         max_iter= self.config.model.max_iter,
                                         class_weight=self.config.model.class_weight)

    def train(self):

        """Compiles and trains the mode and print metricsl"""
        LR_model= self.model.fit(self.X_tr,                                 self.y_tr)
        y_pred = LR_model.predict(self.X_te)
        print("The Precision Score is  : ", precision_score(self.y_te, y_pred))
        print("The Recall Score Is: ",recall_score(self.y_te, y_pred))
        print("The ROC_AUC Score Is: ",roc_auc_score(self.y_te, y_pred))
        print("The F1 Score Is: ",f1_score(self.y_te, y_pred))
        print("The Precision Recall Curve Is: ",precision_recall_curve(self.y_te, y_pred))
        print(classification_report(self.y_te, y_pred))
        return



'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  
  y = df.pop('output')
X = df

X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
X.iloc[X_train] # return dataframe train
  
        self.dataset = None
        self.info = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches
        self.val_subsplits = self.config.train.val_subsplits
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0

        self.image_size = self.config.data.image_size
        self.train_dataset = []
        self.test_dataset = []


    def _preprocess_data(self):
        """ Splits into training and test and set training parameters"""
        train = self.dataset['train'].map(self._load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = self.dataset['test'].map(self._load_image_test)

        self.train_dataset = train.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.test_dataset = test.batch(self.batch_size)

        self._set_training_parameters()

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.train_length = self.info.splits['train'].num_examples
        self.steps_per_epoch = self.train_length // self.batch_size
        self.validation_steps = self.info.splits['test'].num_examples // self.batch_size // self.val_subsplits

    def _normalize(self, input_image, input_mask):
        """ Normalise input image
        Args:
            input_image (tf.image): The input image
            input_mask (int): The image mask

        Returns:
            input_image (tf.image): The normalized input image
            input_mask (int): The new image mask
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    @tf.function
    def _load_image_train(self, datapoint):
        """ Loads and preprocess  a single training image """
        input_image = tf.image.resize(datapoint['image'], (self.image_size, self.image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.image_size, self.image_size))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = self._normalize(input_image, input_mask)

        return input_image, input_mask

    def _load_image_test(self, datapoint):
        """ Loads and preprocess a single test images"""

        input_image = tf.image.resize(datapoint['image'], (self.image_size, self.image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.image_size, self.image_size))

        input_image, input_mask = self._normalize(input_image, input_mask)

        return input_image, input_mask

    def build(self):
        """ Builds the Keras model based """
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [self.base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=layers)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(self.config.model.up_stack.layer_1, self.config.model.up_stack.kernels),  # 4x4 -> 8x8
            pix2pix.upsample(self.config.model.up_stack.layer_2, self.config.model.up_stack.kernels),  # 8x8 -> 16x16
            pix2pix.upsample(self.config.model.up_stack.layer_3, self.config.model.up_stack.kernels),  # 16x16 -> 32x32
            pix2pix.upsample(self.config.model.up_stack.layer_4, self.config.model.up_stack.kernels),  # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=self.config.model.input)
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            self.output_channels, self.config.model.up_stack.kernels, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def train(self):
        """Compiles and trains the model"""
        self.model.compile(optimizer=self.config.train.optimizer.type,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=self.config.train.metrics)

        model_history = self.model.fit(self.train_dataset, epochs=self.epoches,
                                       steps_per_epoch=self.steps_per_epoch,
                                       validation_steps=self.validation_steps,
                                       validation_data=self.test_dataset)

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        predictions = []
        for image, mask in self.dataset.take(1):
            predictions.append(self.model.predict(image))

        return predictions


    def distributed_train(self):
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        with mirrored_strategy.scope():
            self.model = tf.keras.Model(inputs=inputs, outputs=x)
            self.model.compile(...)
            self.model.fit(...)


        os.environ["TF_CONFIG"] = json.dumps(
            {
                "cluster":{
                    "worker": ["host1:port", "host2:port", "host3:port"]
                },
                "task":{
                     "type": "worker",
                     "index": 1
                }
            }
        )

        multi_worker_mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with multi_worker_mirrored_strategy.scope():
            self.model = tf.keras.Model(inputs=inputs, outputs=x)
            self.model.compile(...)
            self.model.fit(...)

        parameter_server_strategy = tf.distribute.experimental.ParameterServerStrategy()

        os.environ["TF_CONFIG"] = json.dumps(
            {
                "cluster": {
                    "worker": ["host1:port", "host2:port", "host3:port"],
                    "ps":  ["host4:port", "host5:port"]
                },
                "task": {
                    "type": "worker",
                    "index": 1
                }
            }
        )
'''