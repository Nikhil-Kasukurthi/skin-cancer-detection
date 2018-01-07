import pandas as pd
import os
import pickle
import numpy as np

def class_dir(parent_path):
    '''
            :param parent_path: The parent path in which the subdirs need to be made
    '''
    if not os.path.exists(os.path.join(parent_path, 'melanoma')):
        os.makedirs(os.path.join(parent_path, 'melanoma'))

    if not os.path.exists(os.path.join(parent_path, 'seborrheic_keratosis')):
        os.makedirs(os.path.join(parent_path, 'seborrheic_keratosis'))

    if not os.path.exists(os.path.join(parent_path, 'nevus')):
        os.makedirs(os.path.join(parent_path, 'nevus'))

train_df = pd.read_csv('ISIC-2017_Training_Part3_GroundTruth.csv')
validation_df = pd.read_csv('ISIC-2017_Validation_Part3_GroundTruth.csv')
test_df = pd.read_csv('ISIC-2017_Test_v2_Part3_GroundTruth.csv')

#train_df = train_df.drop(train_df.index[0])

print(train_df.head())
train_dir = './ISIC-2017_Training_Data'
validation_dir = './ISIC-2017_Validation_Data'
test_dir = './ISIC-2017_Test_v2_Data'

class_dir('./train')
class_dir('./validation')
class_dir('./test')

labels_train = np.zeros((len(train_df), 3))
labels_validation = np.zeros((len(validation_df), 3))
labels_test = np.zeros((len(test_df), 3))


def move_image(type_, dataframe, directory):

    for index, image in enumerate(dataframe['image_id']):
        image_path = os.path.join(directory, image + '.jpg')

        if dataframe.loc[dataframe['image_id'] == image]['melanoma'].all() == 1:
            os.rename(image_path, os.path.join(
                type_, 'melanoma', image + '.jpg'))
            if(type_ == 'train'):
                labels_train[index][0] = 1
            elif(type_ == 'validation'):
                labels_validation[index][0] = 1
            else:
                labels_test[index][0] = 1
        elif dataframe.loc[dataframe['image_id'] == image]['seborrheic_keratosis'].all() == 1:
            os.rename(image_path, os.path.join(
                type_, 'seborrheic_keratosis', image + '.jpg'))
            if(type_ == 'train'):
                labels_train[index][1] = 1
            elif(type_=='validation'):
                labels_validation[index][1] = 1
            else:
            	labels_test[index][1] = 1
        else:
            os.rename(image_path, './' + type_ + '/nevus/' + image + '.jpg')
            if(type_ == 'train'):
                labels_train[index][2] = 1
            elif(type_=='validation'):
                labels_validation[index][2] = 1
            else:
            	labels_test[index][2] = 1

move_image('train', train_df, train_dir)
move_image('validation', validation_df, validation_dir)
move_image('test', test_df, test_dir)

pickle.dump([labels_train, labels_validation, labels_test], open('labels.pkl','wb'))
