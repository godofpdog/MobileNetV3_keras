import os 
import sys 
import logging
import argparse
import configparser
import numpy as np
from keras.models import load_model
from src.generator import DataGenerator
from src.learning_rate_schedule import learning_rate_scheduler
from src.MobileNet_V3 import build_mobilenet_v3
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, 
                             LearningRateScheduler, 
                             ReduceLROnPlateau, 
                             EarlyStopping)

logging.basicConfig(level=logging.INFO)

def _main(args):

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** MobileNet V3 configuration
    input_size = config_client.getint('model', 'input_size')
    model_size = config_client.get('model', 'model_size')
    pooling_type = config_client.get('model', 'pooling_type')
    num_classes = config_client.getint('model', 'num_classes')

    # ** training configuration
    epochs = config_client.getint('train', 'epochs')
    batch_size = config_client.getint('train', 'batch_size')
    save_path = config_client.get('train', 'save_path')

    # ** Dataset 
    train_directory = config_client.get('data', 'train')
    valid_directory = config_client.get('data', 'valid')

    # ** initialize data generators
    train_generator = DataGenerator(dir_path=train_directory, batch_size=batch_size, aug_freq=0, image_size=input_size)
    valid_generator = DataGenerator(dir_path=valid_directory, batch_size=batch_size, aug_freq=0, image_size=input_size)

    # ** initalize model
    try:
        model = load_model(os.path.join(ROOT_DIR, onfig_client.get('train', 'pretrained_path')))
    except Exception as e:
        logging.info('Failed to load pre-trained model.')
        model = build_mobilenet_v3(input_size, num_classes, model_size, pooling_type)

    model.compile(optimizer=Adam(lr=3e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # ** setup keras callback
    filename = 'ep{epoch:03d}-loss{loss:.3f}.h5'
    weights_directory = os.path.join(ROOT_DIR, 'weights')
    save_path = os.path.join(weights_directory, filename)
    checkpoint = ModelCheckpoint(save_path, monitor='loss', save_best_only=True, period=5)
    scheduler = LearningRateScheduler(learning_rate_scheduler)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)

    # ** start training
    model.fit_generator(generator       = train_generator, 
                        epochs          = epochs,
                        callbacks       = [checkpoint, scheduler],
                        )

    model.save(os.path.join(ROOT_DIR, save_path))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main(args)

