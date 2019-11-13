#!/usr/bin/env python3

# # Histopathologic Cancer Detection
# #### Aim: Create algorithm to identify metastatic cancer from small image
# patches taken from the larger digital pathology scans.
# #### These images are taken from the modified version of PathCamelon (PCam)
# benchmark dataset (duplicate entries have been removed, NB: duplicate entries
#  existed owing to probabilistic sampling)
#
#  Data description: Dataset contains large number of small pathology images to
#  classify. Files are named with an image "id". The train_labels.csv file
#  provides the ground truh for the images in the train folder.
#  We are predicting labels for the images in the test folder.
#  A positive label indicates that the center 32x32px region of a patch contains
#  atleast 1 pixel of tumor tissue. Tumor tissue in the outer region of the
#  patch does not influence the label. The outer region is provided to enable
#  fully convolutional models which do not use padding to ensure consistend
#  behaviour when applied to the whole slide.
#
###############################################################################
###############################################################################
#               Required Imports
#
import matplotlib as mp
mp.use('Agg')
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import PIL
import random  # to generate psuedo random numbers
from imutils import paths  # A series of convenience functions to
from subprocess import check_output
import h5py
import shutil  # copy/remove on files and collection of files
import json
import zipfile  # read ZIP file
from glob import glob
import pandas as pd  # dataprocessing
import numpy as np  # linear algebra
import sys
import os  # miscellaneous operating system interfaces
import argparse
import matplotlib.pyplot as pyplot
###############################################################################
###############################################################################
#               Define Functions
#


def img_load_array(path):
    """
Convert loaded image to an numpy array

Prereq: numpy, from keras_preprocessing.image import img_to_array, load_img

Argument: path (dtype: string)

Return:Image as an numpy array
    """
    # Read in the current image
    image_keras = load_img(path)
    # Convert image to numpy array
    image_array = img_to_array(image_keras)
    return image_array


def get_latest_checkpoint(checkpoint_dir):
    '''
Return the latest checkpoint index and file in user provided directory
    '''
    all_files = {}
    file_list = glob(os.path.join(checkpoint_dir, 'saved-model*.hdf5'))
    for file in file_list:
        all_files[(int(file.split('/')[-1].split('saved-model-')[-1][0:2]))] = file

    return max(all_files.keys()), all_files[max(all_files.keys())]


###############################################################################
###############################################################################
#               Argument Parsing
#
parser = argparse.ArgumentParser(description='Train CNN for input data')
parser.add_argument('--split-fraction-validation', dest='VAL_SPLIT', type=float,
                    default=0.2, help='validation dataset split fraction (default: 0.2)')
parser.add_argument('--number-of-epochs', dest='NUM_EPOCHS', type=int,
                    default=20, help='number of epochs(default:20)')
parser.add_argument('--initial-learning-rate', dest='INIT_LR', type=float,
                    default=1e-3, help='initial-learning-rate (default: 1e-3)')
parser.add_argument('--batch-size', dest='BATCH_SIZE', type=int,
                    default=32, help='batch size (default: 32)')
parser.add_argument('--image-height', dest='HEIGHT', type=int,
                    default=224, help='Image height in pixels (default: 224)')
parser.add_argument('--image-width', dest='WIDTH', type=int,
                    default=224, help='Image width in pixels (default: 224)')
parser.add_argument('--default-image-height', dest='DEF_HEIGHT', type=int,
                    default=96, help='Image height in pixels (default: 96)')
parser.add_argument('--default-image-width', dest='DEF_WIDTH', type=int,
                    default=96, help='Image width in pixels (default: 96)')
parser.add_argument('--dropout-rate', dest='DROPOUT_RATE', type=float,
                    default=0.5, help='Dropout rate (default: 0.5)')
parser.add_argument('--resuming', action='store_true',
                    default=False, help='Resume from checkpoint (default: False)')
parser.add_argument('--verbose', action='store_true',
                    default=False, help='Verbose output (default: False)')
parser.add_argument('--run-name', dest='run_name', type=str, default='',
                    help = 'create a subfolder with run_name' )
parser.add_argument('--user-defined-checkpoint', dest='user_defined_checkpoint',
                    type=str, default=False,
                    help = 'Resume from previous checkpoint or user define one  (default: False) ' )
parser.add_argument('--checkpoint-file-path', dest='checkpoint_file_path',type=str,
                    default='', help='Enter the checkpoint file path (default: '')')
args = parser.parse_args()


###############################################################################
###############################################################################
# Set paths that you will use in this notebook
base_dir = os.path.join('/home/bithika/ml/Kaggle/pcam')
base_dir = os.path.join(base_dir, args.run_name)
input_dir = os.path.join(base_dir, 'input')
checkpoint_dir = os.path.join(base_dir, 'checkpoints')
plot_dir = os.path.join(base_dir, 'plots')
output_dir = os.path.join(base_dir, 'output')
os.system("mkdir -p {} {} {} {} {}".format(base_dir,
                                           input_dir, checkpoint_dir,
                                           plot_dir, output_dir))
if args.verbose:
    print("Made required directories...")
    sys.stdout.flush()

###############################################################################
###############################################################################
# ## Explore training data
# training, validation, and testing directories
VALTRAIN_PATH = os.path.sep.join([input_dir, "VALTRAIN"])
TEST_PATH = os.path.sep.join([input_dir, "TEST"])

# grab the paths to all the input images in the TRAIN directory
# and shuffle them
valtrainPaths = list(paths.list_images(VALTRAIN_PATH))
random.seed(42)
random.shuffle(valtrainPaths)
if args.verbose:
    print("Set paths and constants...")
    sys.stdout.flush()

#  we need to make a dataframe contanint the every training and validation
# example image's path, id and labels
# SAVE train_labels.csv as a dataframe
labels = pd.read_csv(os.path.join(input_dir, 'train_labels.csv'))

# now we want to add filepath's to a dataframe which also contains id and label
# we use glob. Globbing is a technical term for matching files by name or type
# of file.
# here we use it to match by file type.
# we need to go into subfolders "oldtrain" inside the VAL_TRAIN_PATH -
# this should be created before we split data

# join the folder
val_train_tif = os.path.join(VALTRAIN_PATH, "oldtrain")

# using globbing to match all files with .tif file type and save in a dataframe
path_labels = pd.DataFrame(
    {'path': glob(os.path.join(val_train_tif, '*.tif'))})

path_labels.head()  # glance at the data


# we add another column labeled 'id'  to the path_labels dataframe
# it's value is just the file name of the tif file saved in a specific filepath
path_labels['id'] = path_labels.path.map(
    lambda x: ((x.split('n/')[1].split('.')[0])))

# merge dataframes on label = 'id'
path_labels = path_labels.merge(labels, on='id')


###############################################################################
# Choose 6 random positive and negative examples, find their respective
# path and then display them in subplot
positive_indices = list(np.where(path_labels["label"])[0])
negative_indices = list(np.where(path_labels["label"] == False)[0])

# take 3 random positive indices and negative indices
random_positive_indices = random.sample(positive_indices, 3)
random_negative_indices = random.sample(negative_indices, 3)

if not args.resuming:
    if args.verbose:
        print("Plotting preview of data")
        sys.stdout.flush()
    # Make preview figure
    fig, ax = pyplot.subplots(2, 3, figsize=(20, 10))  # 2 rows and 3 columns
    # Add a centered title to the figure.
    fig.suptitle(
        'Histopathologic scans of lymph node sections',
        fontsize=20,
        fontweight='bold')
    for i in range(0, 3):
        # Display image in first row and i'th column axes location - 1/255 ensures
        # that RGB values are in range [0,1] for float values
        ax[0, i].imshow(img_load_array(
            path_labels.iloc[random_positive_indices[i], 0]) / 255)
        # Set a  centered title for the image diplayed at axes defined above
        ax[0, i].set_title("Positive example", fontweight='bold')

        # Display image in first row and i'th column axes location - 1/255 ensures
        # that RGB values are in range [0,1] for float values
        ax[1, i].imshow(img_load_array(
            path_labels.iloc[random_negative_indices[i], 0]) / 255)
        # Set a  centered title for the image diplayed at axes defined above
        ax[1, i].set_title("Negative example", fontweight='bold')
        pyplot.savefig(
            os.path.join(
                plot_dir,
                "histpath_label_img.png"),
            bbox_inches='tight')


###############################################################################
###############################################################################
# ## Split train data into training and validation datasets
if args.verbose:
    print('spliting train data into training and validation datasets')
    sys.stdout.flush()

# Classic way with ```train_test_split```
TestPaths = list(paths.list_images(TEST_PATH))  # list files inside
ValTrainPaths = list(paths.list_images(VALTRAIN_PATH))  # list files inside
if args.verbose:
    print("There are " + str(len(ValTrainPaths)) + " training examples.")
    print("There are " + str(len(TestPaths)) + " test examples.")
    sys.stdout.flush()

# Split some-% of training dataset into validation dataset
path_labels['label'] = path_labels['label'].astype(
    str)  # cast label values as str dtype
TrainPaths, ValidPaths = train_test_split(
    path_labels, test_size=args.VAL_SPLIT, random_state=220, stratify=path_labels['label'])  # split pandas
# dataframe 20% of valtrain data is saved as validation data using the random seed defined
# above and stratified by 'labels' in panda dataframe


###############################################################################
###############################################################################
# Data augmentation using data from train_test_split function
if args.verbose:
    print('starting training and validation dataset augmentation')
    sys.stdout.flush()
# initialize the training training data augmentation object
# rescale
TrainAugmentation = ImageDataGenerator(
    # include? preprocessing_function=preprocess_input only if using keras
    # inbuilt resnet
    rescale=1 / 255.0,  # normalisation: scaling data to the range
    # of 0-1
    rotation_range=90,  # degrees
    # Horizontal and Vertical Shift Augmentation
    width_shift_range=0.1,  # fraction of total width
    height_shift_range=0.1,  # fraction of total height
    zoom_range=0.2,  # i.e zoom range is [0.95,1.05]
    # not sure about keeping 0.05
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True
)

# fill_mode="nearest"

###############################################################################
###############################################################################
# initialize the validation (and testing) data augmentation object
Val_Test_Augmentation = ImageDataGenerator(rescale=1 / 255.0)


###############################################################################
###############################################################################
TrainGenerator = TrainAugmentation.flow_from_dataframe(
    dataframe=TrainPaths,
    directory=None,
    x_col='path',
    y_col='label',
    color_mode="rgb",
    target_size=(
        args.HEIGHT,
        args.WIDTH),
    class_mode="binary",
    batch_size=args.BATCH_SIZE,
    seed=10986,
    shuffle=True)
# could use interpolation = "bicubic" if I intend to crop the images


###############################################################################
###############################################################################
ValidGenerator = Val_Test_Augmentation.flow_from_dataframe(
    dataframe=ValidPaths,
    directory=None,
    x_col='path',
    y_col='label',
    target_size=(
        args.HEIGHT,
        args.WIDTH),
    color_mode="rgb",
    class_mode="binary",
    batch_size=args.BATCH_SIZE,
    shuffle=False)  # shuffling is

###############################################################################
###############################################################################
# Creating or Loading the model
#
# This CNN has two parts:
# 1. **Convolutional base:** which is composed by a stack of convolutional and
# pooling layers. The main goal of the convolutional base is to generate
# features from the image. For an intuitive explanation of convolutional and
#  pooling layers, please refer to Chollet (2017)
# 2. **Classifier:** which is usually composed by fully connected layers.
# The main goal of the classifier is to classify the image based on the
# detected features. A fully connected layer is a layer whose neurons have
# full connections to all activation in the previous laye
if args.verbose:
    print('creating model')
    sys.stdout.flush()

epoch_at_start = 1
if args.resuming:
    # load model from last checkpoint
    if args.user_defined_checkpoint and args.checkpoint_file_path != '':
        # ask user to enter chekpoint file with full path 
        checkpoint_file = args.checkpoint_file_path
    else:
        epoch_at_start, checkpoint_file = get_latest_checkpoint(checkpoint_dir)
    model = load_model(checkpoint_file)
else:
    # load pretrained ResNet-50
    if args.verbose:
        print('loading convolutional base part1: ResNet50')
        sys.stdout.flush()
    # Convolutional base, which performs feature extraction.
    ConvolutionalBase = ResNet50(include_top=False,
                                 weights='imagenet',
                                 input_shape=(args.HEIGHT, args.WIDTH, 3))
    # Classifier: Use fully connected layers. In this classifier we add a stack of
    # fully-connected layers which is fed by features extracted from the convolutional
    # base
    model = Sequential()  # crease a sqequntial model
    # add layers
    model.add(ConvolutionalBase)
    model.add(Flatten())  # A tensor, reshaped into 1-D
    # regular densely-connected NN layer with
    model.add(Dense(256, use_bias=False))
    # dimensions 256
    model.add(BatchNormalization())
    # Applies an activation function to an output,
    model.add(Activation("relu"))
    # retains shape
    # this core layer randomly sets a fraction of
    model.add(Dropout(args.DROPOUT_RATE))
    # input units to
    # 0 at each update during training time, which helps prevent overfitting
    model.add(Dense(1, activation='sigmoid'))
    # Now we need to train the last few layers instead of just the last one.
    # This is because of the size-similarity matrix, since our task is to identify
    # cancer cells Imagenet can't be considered a similar dataset.  In principle as
    #  we have a large dataset we can train the model from scratch but it can be
    #  useful to initialize the model from a pretrained model, using its
    # architecture and weights. We use ResNet50 ```keras``` implementation for this.
    #
    # We start by retraining, 143th layer of ResNet50 i.e 'res5a_branch2a'
    set_trainable = False
    for layer in ConvolutionalBase.layers:
        if layer.name == 'res5a_branch2a':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

if args.verbose:
    print(model.summary())
    sys.stdout.flush()


###############################################################################
###############################################################################
# compile the model
if args.verbose:
    print('compile model')
    sys.stdout.flush()
model.compile(optimizers.Adam(args.INIT_LR), loss="binary_crossentropy",
              metrics=["accuracy"])

if args.verbose:
    print('begin training')
    sys.stdout.flush()

###############################################################################
###############################################################################
# Define callback functions
###############################################################################
# Stop training when a monitored quantity has stopped improving.
# early_stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=2,
#                              restore_best_weights=True)
# Reduce learning rate when a metric has stopped improving.
reduce = ReduceLROnPlateau(
    monitor='val_loss',
    patience=1,
    verbose=1,
    factor=0.1)


# checkpoint
# IMP for userdefined callbacks:
# alternative solution is to pickle the callback instance every time we save a
# checkpoint, then we can load this pickle when resuming and reconstruct the original
# callback with all its correct values.
filepath = os.path.join(
    checkpoint_dir,
    "saved-model-{epoch:02d}.hdf5")
# Setting 'save_weights_only' to False in the Keras callback 'ModelCheckpoint' will
# save the full model
if args.verbose:
    print('beging checkpointing')
    sys.stdout.flush()
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                             verbose=0, save_best_only=False,
                             save_weights_only=False, mode='auto', period=1)

callback_list = [reduce, checkpoint] # early_stop removed

###############################################################################
###############################################################################
#                        TRAIN
Train_step_size = TrainGenerator.n // TrainGenerator.batch_size
Valid_step_size = ValidGenerator.n // ValidGenerator.batch_size

history = model.fit_generator(TrainGenerator,
                              steps_per_epoch=Train_step_size,
                              epochs=args.NUM_EPOCHS, verbose=1,
                              callbacks=callback_list,
                              validation_data=ValidGenerator,
                              validation_steps=Valid_step_size,
                              use_multiprocessing=True,
                              initial_epoch=epoch_at_start)


###############################################################################
###############################################################################
# ## Post training:
epochs = [i for i in range(1, len(history.history['loss']) + 1)]

# Plot training graphs to ascertain performance via accuracies and losses
# varied over epochs
pyplot.plot(
    epochs,
    history.history['loss'],
    color='blue',
    label="training_loss")
pyplot.plot(
    epochs,
    history.history['val_loss'],
    color='red',
    label="validation_loss")
pyplot.legend(loc='best')
pyplot.title('training')
pyplot.xlabel('epoch')
pyplot.savefig(os.path.join(plot_dir, "training.png"), bbox_inches='tight')

pyplot.plot(
    epochs,
    history.history['acc'],
    color='blue',
    label="training_accuracy")
pyplot.plot(
    epochs,
    history.history['val_acc'],
    color='red',
    label="validation_accuracy")
pyplot.legend(loc='best')
pyplot.title('validation')
pyplot.xlabel('epoch')
pyplot.savefig(os.path.join(plot_dir, "validation.png"), bbox_inches='tight')

# ROC Plot
roc_validation_generator = ImageDataGenerator(
    rescale=1. /
    255).flow_from_dataframe(
        ValidPaths,
        x_col='path',
        y_col='label',
        target_size=(
            args.HEIGHT,
            args.WIDTH),
    class_mode='binary',
    batch_size=args.BATCH_SIZE,
    shuffle=False)

if args.verbose:
    print('calculate accuracy, losses and ROC')
    sys.stdout.flush()
predictions = model.predict_generator(
    roc_validation_generator,
    steps=len(roc_validation_generator),
    verbose=2)
false_positive_rate, true_positive_rate, threshold = roc_curve(
    roc_validation_generator.classes, predictions)
area_under_curve = auc(false_positive_rate, true_positive_rate)

pyplot.plot([0, 1], [0, 1], 'k--')
pyplot.plot(false_positive_rate, true_positive_rate,
            label='AUC = {:.3f}'.format(area_under_curve))
pyplot.xlabel('False positive rate')
pyplot.ylabel('True positive rate')
pyplot.title('ROC curve')
pyplot.legend(loc='best')
pyplot.savefig(os.path.join(plot_dir, 'ROC_PLOT.png'), bbox_inches='tight')


###############################################################################
###############################################################################
# ## Predictions
test_tif = os.path.join(TEST_PATH, 'test')
Test_df = pd.DataFrame({'path': glob(os.path.join(test_tif, '*.tif'))})
Test_df['id'] = Test_df.path.map(lambda x: (x.split('t/')[1].split('.')[0]))
if args.verbose:
    print(Test_df.head(3))
    sys.stdout.flush()

TestDatagen = ImageDataGenerator(rescale=1. / 255)

TestGenerator = TestDatagen.flow_from_dataframe(dataframe=Test_df,
                                                directory=None,
                                                x_col='path',
                                                target_size=(
                                                    args.HEIGHT, args.WIDTH),
                                                class_mode=None,
                                                batch_size=1,
                                                shuffle=False)
tta_steps = 5
submission = pd.DataFrame()

if args.verbose:
    print('CNN on test data')
    sys.stdout.flush()

for index in range(0, len(Test_df)):
    data_frame = pd.DataFrame({'path': Test_df.iloc[index, 0]}, index=[index])
    data_frame['id'] = data_frame.path.map(
        lambda x: ((x.split('t/')[2].split('.')[0])))
    img_path = data_frame.iloc[0, 0]
    test_img = cv2.imread(img_path)
    test_img = cv2.resize(test_img, (args.HEIGHT, args.WIDTH))
    test_img = np.expand_dims(test_img, axis=0)
    predictionsTTA = []
    for i in range(0, tta_steps):
        preds = model.predict_generator(
            TestDatagen.flow_from_dataframe(
                dataframe=data_frame,
                directory=None,
                x_col='path',
                target_size=(
                    args.HEIGHT,
                    args.WIDTH),
                class_mode=None,
                batch_size=1,
                shuffle=False),
            steps=1)
        predictionsTTA.append(preds)
    clear_output()
    prediction_entry = np.array(np.mean(predictionsTTA, axis=0))
    data_frame['label'] = prediction_entry
    submission = pd.concat([submission, data_frame[['id', 'label']]])

###############################################################################
###############################################################################
#                     Prepare Submission
if args.verbose:
    print('saving submission')
    sys.stdout.flush()
submission.set_index('id')
submission.to_csv(os.path.join(output_dir, 'submission_early_stop_removed.csv'),
                  index=False, header=True)
