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
#
#  Run the following cell to run all the packages and dependencies

# ## Packages and other dependencies



### load required packages ###
import matplotlib.pyplot as pyplot
import os  # miscellaneous operating system interfaces
import numpy as np  # linear algebra
import pandas as pd  # dataprocessing
from glob import glob
import zipfile  # read ZIP file
import json
import shutil  # copy/remove on files and collection of files
# (high-level file and directory handling)

# needed to checkpoint: specifically to output network weights in HDF5 format.
import h5py
# look at the data files
from subprocess import check_output
from imutils import paths  # A series of convenience functions to
# make basic image processing functions such as translation,
# rotation, resizing, skeletonization, displaying Matplotlib
# images, sorting contours, detecting edges, and much more easier
# with OpenCV and both Python 2.7 and Python 3.
import random  # to generate psuedo random numbers
import PIL

# import function to split arrays into random train and test subsets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
# Checkpoint the weights when validation accuracy improves
from keras.callbacks import ModelCheckpoint
# Set paths that you will use in this notebook
base_dir = os.path.join('/home/bithika/ml/Kaggle/pcam/')
input_dir = os.path.join(base_dir, 'input')
checkpoint_dir = os.path.join(base_dir, 'checkpoints')
plot_dir = os.path.join(base_dir, 'plots')
output_dir = os.path.join(base_dir, 'output')
os.system("mkdir -p {} {} {} {}".format(base_dir,
                                        input_dir,checkpoint_dir,
                                        plot_dir, output_dir))


# ## Define Constants

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = .125
# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 5  # training iterations
INIT_LR = 1e-3  # initial learning rate
BATCH_SIZE = 32  # batch size of 32 is good for CPUs, take 64 for GPUs
(HEIGHT, WIDTH) = (224, 224)  # rescale images (higher performance; crop images)
(DEF_HEIGHT, DEF_WIDTH) = (96, 96)  # used for test data
DROPOUT_RATE = 0.5


# ## Explore training data
# training, validation, and testing directories
VALTRAIN_PATH = os.path.sep.join([base_dir, "VALTRAIN"])
TEST_PATH = os.path.sep.join([base_dir, "TEST"])


# grab the paths to all the input images in the TRAIN directory
# and shuffle them
valtrainPaths = list(paths.list_images(VALTRAIN_PATH))
random.seed(42)
random.shuffle(valtrainPaths)


#  we need to make a dataframe contanint the every training and validation
# example image's path, id and labels
# SAVE train_labels.csv as a dataframe
labels = pd.read_csv(os.path.join(base_dir, 'train_labels.csv'))

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


# Choose 6 random positive and negative examples, find their respective
# path and then display them in subplot

positive_indices = list(np.where(path_labels["label"])[0])
negative_indices = list(np.where(path_labels["label"] == False)[0])

# take 3 random positive indices and negative indices
random_positive_indices = random.sample(positive_indices, 3)
random_negative_indices = random.sample(negative_indices, 3)


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


# ## Split train data into training and validation datasets

# Classic way with ```train_test_split```

TestPaths = list(paths.list_images(TEST_PATH))  # list files inside
ValTrainPaths = list(paths.list_images(VALTRAIN_PATH))  # list files inside
print("There are " + str(len(ValTrainPaths)) + " training examples.")
print("There are " + str(len(TestPaths)) + " test examples.")


# Split 20% of training dataset into validation dataset
#


path_labels['label'] = path_labels['label'].astype(
    str)  # cast label values as str dtype
TrainPaths, ValidPaths = train_test_split(
    path_labels, test_size=0.2, random_state=220, stratify=path_labels['label'])  # split pandas
# dataframe 20% of valtrain data is saved as validation data using the random seed defined
# above and stratified by 'labels' in panda dataframe



# ## Data augmentation using data from train_test_split function

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

# initialize the validation (and testing) data augmentation object
Val_Test_Augmentation = ImageDataGenerator(rescale=1 / 255.0)


TrainGenerator = TrainAugmentation.flow_from_dataframe(
    dataframe=TrainPaths,
    directory=None,
    x_col='path',
    y_col='label',
    color_mode="rgb",
    target_size=(
        HEIGHT,
        WIDTH),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    seed=10986,
    shuffle=True)
# could use interpolation = "bicubic" if I intend to crop the images


ValidGenerator = Val_Test_Augmentation.flow_from_dataframe(
    dataframe=ValidPaths,
    directory=None,
    x_col='path',
    y_col='label',
    target_size=(
        HEIGHT,
        WIDTH),
    color_mode="rgb",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False)  # shuffling is


# ## Creating the model
#
# CNN has two parts:
# 1. **Convolutional base:** which is composed by a stack of convolutional and
# pooling layers. The main goal of the convolutional base is to generate
# features from the image. For an intuitive explanation of convolutional and
#  pooling layers, please refer to Chollet (2017)
# 2. **Classifier:** which is usually composed by fully connected layers.
# The main goal of the classifier is to classify the image based on the
# detected features. A fully connected layer is a layer whose neurons have
# full connections to all activation in the previous laye

# load pretrained ResNet-50
# Convolutional base, which performs feature extraction.
ConvolutionalBase = ResNet50(include_top=False,
                             weights='imagenet',
                             input_shape=(HEIGHT, WIDTH, 3))


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
model.add(Activation("relu"))  # Applies an activation function to an output,
# retains shape
model.add(Dropout(DROPOUT_RATE))  # this core layer randomly sets a fraction of
# input units to
# 0 at each update during training time, which helps prevent overfitting
model.add(Dense(1, activation='sigmoid'))
model.summary()


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


# compile the model
model.compile(optimizers.Adam(INIT_LR), loss="binary_crossentropy",
              metrics=["accuracy"])


# ## Train model

Train_step_size = TrainGenerator.n // TrainGenerator.batch_size
Valid_step_size = ValidGenerator.n // ValidGenerator.batch_size

##############################################################################
###################   callback function ######################################
##############################################################################
# Stop training when a monitored quantity has stopped improving.
early_stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=2,
                              restore_best_weights=True)
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
    "saved-model-{epoch:02d}-{val_acc:.3f}.hdf5")
# Setting 'save_weights_only' to False in the Keras callback 'ModelCheckpoint' will
# save the full model
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                             verbose=0, save_best_only=False,
                             save_weights_only=False, mode='auto', period=1)

callback_list = [early_stopper, reduce, checkpoint]

history = model.fit_generator(TrainGenerator,
                              steps_per_epoch=Train_step_size,
                              epochs=NUM_EPOCHS, verbose=1,
                              callbacks=callback_list,
                              validation_data=ValidGenerator,
                              validation_steps=Valid_step_size,
                              use_multiprocessing=False)


# ## Post training:
# Plot training graphs to ascertain performance via accuracies and losses varied over epochs
#


epochs = [i for i in range(1, len(history.history['loss']) + 1)]

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
pyplot.savefig(os.path.join(plot_dir,"training.png"), bbox_inches='tight')


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
pyplot.savefig(os.path.join(plot_dir,"validation.png"), bbox_inches='tight')



# ROC Plot

roc_validation_generator = ImageDataGenerator(
    rescale=1. /
    255).flow_from_dataframe(
        ValidPaths,
        x_col='path',
        y_col='label',
        target_size=(
            HEIGHT,
            WIDTH),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False)

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
pyplot.savefig(os.path.join(plot_dir,'ROC_PLOT.png'), bbox_inches='tight')



# ## Predictions
test_tif = os.path.join(TEST_PATH, 'test')
Test_df = pd.DataFrame({'path': glob(os.path.join(test_tif, '*.tif'))})
Test_df['id'] = Test_df.path.map(lambda x: (x.split('t/')[1].split('.')[0]))
Test_df.head()

TestDatagen = ImageDataGenerator(rescale=1. / 255)

TestGenerator = TestDatagen.flow_from_dataframe(dataframe=Test_df,
                                                directory=None,
                                                x_col='path',
                                                target_size=(HEIGHT, WIDTH),
                                                class_mode=None,
                                                batch_size=1,
                                                shuffle=False)
tta_steps = 5
submission = pd.DataFrame()

for index in range(0, len(Test_df)):
    data_frame = pd.DataFrame({'path': Test_df.iloc[index, 0]}, index=[index])
    data_frame['id'] = data_frame.path.map(
        lambda x : ((x.split('t/')[1].split('.')[0])))
    img_path = data_frame.iloc[0, 0]
    test_img = cv2.imread(img_path)
    test_img = cv2.resize(test_img, (HEIGHT, WIDTH))
    test_img = np.expand_dims(test_img, axis=0)
    predictionsTTA = []
    for i in range(0, tta_steps):
        preds = model.predict_generator(
            TestDatagen.flow_from_dataframe(
                dataframe=data_frame,
                directory=None,
                x_col='path',
                target_size=(
                    HEIGHT,
                    WIDTH),
                class_mode=None,
                batch_size=1,
                shuffle=False),
            steps=1)
        predictionsTTA.append(preds)
    clear_output()
    prediction_entry = np.array(np.mean(predictionsTTA, axis=0))
    data_frame['label'] = prediction_entry
    submission = pd.concat([submission, data_frame[['id', 'label']]])



submission.set_index('id')




submission.to_csv(os.path.join(output_dir, 'submission_trial001.csv'),
                  index=False, header=True)
