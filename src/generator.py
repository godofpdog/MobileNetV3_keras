import os
import cv2
import sys
import keras 
import numpy as np  
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder

class DataGenerator(Sequence):
    def __init__(self, dir_path, batch_size=16, aug_freq=0.5, image_size=112, shuffle=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.aug_freq = aug_freq
        self.shuffle = shuffle  
        self.__get_all_paths()
        self.__target_encoding()
        self.__augmentation_operations()
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.paths_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_num_classes(self):
        return self.num_classes

    def __get_all_paths(self):
        if not os.path.exists(self.dir_path):
            raise 'input dir does not exist'
        print('** start getting images/labels path.......')
        self.paths_list = []
        self.label_list = []
        for i, _id in enumerate(os.listdir(self.dir_path)):
            _id_path = os.path.join(self.dir_path, _id)
            for path in os.listdir(_id_path):
                img_path = os.path.join(_id_path, path)
                self.paths_list.append(img_path)
                self.label_list.append(_id)
        self.num_classes = i + 1
        print('** num_classes : ', self.num_classes)
        return None
    
    def __target_encoding(self):
        self.le = LabelEncoder()
        self.le.fit(self.label_list)

    def __len__(self):
        return int(np.floor(len(self.paths_list) / self.batch_size))

    def __getitem__(self, index):
        # ** get batch indices
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.paths_list))
        indices = self.indices[start_index:end_index]

        # ** get batch inputs
        paths_batch_list = [self.paths_list[i] for i in indices]
        label_batch_list = [self.label_list[i] for i in indices]

        # ** generate batch data
        X, y = self.__data_generation(paths_batch_list, label_batch_list)

        return X, y

    def __data_generation(self, paths_batch_list, label_batch_list):
        X = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        y = np.empty((self.batch_size))

        for i, (path, label) in enumerate(zip(paths_batch_list, label_batch_list)):
            X[i, :, :, :] = self.__image_augmentation(cv2.imread(path))
            y[i] = self.le.transform([label])[0]

        y_one_hot = to_categorical(y, self.num_classes)

        return X, y_one_hot

    def __image_augmentation(self, img):
        if img is None:
            raise '** Failed to read image.'
        # to rgb
        img_copy = img.copy()
        img_copy = img_copy[:, :, ::-1]

        # do aug
        if self.aug_freq > np.random.uniform(0, 1, 1):
            img_aug = self.aug_ops.augment_image(img_copy)
        else:
            img_aug = img_copy

        img_norm = self.__normalize(img_aug)      

        return cv2.resize(img_norm, (self.image_size, self.image_size))

    def __augmentation_operations(self):
        self.aug_ops = iaa.Sequential(
            [
                self.__sometimes(iaa.Fliplr(1), 0.5),
                self.__sometimes(iaa.Affine(scale=iap.Uniform(1.0, 1.2).draw_samples(1)), 0.3),
                self.__sometimes(iaa.AdditiveGaussianNoise(scale=0.05*255), 0.2),
                self.__sometimes(iaa.OneOf(
                    [
                        iaa.CropAndPad(percent=(iap.Uniform(0.0, 0.20).draw_samples(1)[0], 
                                                iap.Uniform(0.0, 0.20).draw_samples(1)[0]),
                                       pad_mode=["constant"], 
                                       pad_cval=(0, 128)),
                        iaa.Crop(percent=(iap.Uniform(0.0, 0.15).draw_samples(1)[0], 
                                          iap.Uniform(0.0, 0.15).draw_samples(1)[0]))
                    ]
                )),
                self.__sometimes(iaa.OneOf([
                    iaa.LogContrast(gain=iap.Uniform(0.9, 1.2).draw_samples(1)), 
                    iaa.GammaContrast(gamma=iap.Uniform(1.5, 2.5).draw_samples(1))]))
            ],
            random_order=True       
        )
        return None
    
    def __normalize(self, img):
        return img / 255

    def __sometimes(self, aug, prob=0.5):
        return iaa.Sometimes(prob, aug)