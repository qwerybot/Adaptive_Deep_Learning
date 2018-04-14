from __future__ import division
#import sys
import os
import csv
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
# Import datasets, classifiers and performance metrics
from sklearn import svm
from sklearn.metrics import compare

from math import ceil
import util
from threading import Thread


IMG_SIZE = 228
Percentage_TESTED = 10
#DATA_FILE = 'all_image_features.csv'
#DATA_FILE = 'all_image_features_norm.csv'
##DATA_FILE = 'top-1-few-features.csv'
DATA_FILE = 'all_new_features_hier_norm.csv'
PATH_TO_FILES = '/images/val/images'

list_machines = (
    'nn',
    'dt16',
    'vc'
)


def machines_avialable():
    list_machines_available = []
    for first_level_machine in list_machines:
        for second_level_machine in list_machines:
            for third_level_machine in list_machines:
                list_machines_available.append(str(first_level_machine)+" - " +str(second_level_machine)+" - "+str(third_level_machine))
    
    return list_machines_available
    

def compare_array(list1, list2):
    """
    It compares to arrays and returns how many times they are equal
    """
    success = 0
    for position, number in enumerate(list1):
        if number == list2[position]:
            success = success + 1

    return success, len(list1)-success


def cv_training_data (amount_images):
    """
    This functions returns the data that will be used for training and test different machine learning models.
    This information is collected from the file DATA_FILE.
    Return:
        data: Data for training the models
        data_result: data for validation
    """

    data = []
    first_level = []
    second_level = []
    third_level = []

    # Getting the images for training and testing
    row_count = 0
    with open(DATA_FILE, 'rb') as csvfile:
        lines = [line.decode('utf-8-sig') for line in csvfile]

        for row in csv.reader(lines):
            # Remove the headers of csv file
            if row_count is 0:
                row_count = row_count + 1
                continue

            data.append(row[-7:])
            first_level.append((row[0],row[1]))
            second_level.append((row[0],row[2]))
            third_level.append((row[0],row[3]))
            row_count = row_count + 1
            if row_count > amount_images:
                break
                
    return  data, first_level, second_level, third_level

# Nearest Neighbours model - TRAINING and PREDICTION
def nearest_neighbour(X_train, X_test, Y_test):
    """
    Nearest neighbour function that returns the prediction of a list of images. With K = 5
    Args:
        X_train: List of images features used for training
        X_test: List of images results used for validate the trained images.
        Y_train: List of images features predicted
    """
    # Create and fit a nearest-neighbor classifier
    knn = KNeighborsClassifier()
    
    knn.fit(X_train, X_test)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')

    # Prediction
    predicted = knn.predict(Y_test)

    return predicted

# Decision tree of level 2, 5,8,12 and 16 - TRAINING and PREDICTION
def decision_tree(X_train, X_test, Y_train):
    """
    Decision Tree function that returns the prediction of a list of images. This function allows different deepth levels: 2,5,8,12 and 16
    Args:
        X_train: List of images features used for training
        X_test: List of images results used for validate the trained images.
        Y_train: List of images features predicted
    """

    # Create tree
    regr_2 = DecisionTreeRegressor(max_depth=2)
    regr_5 = DecisionTreeRegressor(max_depth=5)
    regr_8 = DecisionTreeRegressor(max_depth=8)
    regr_12 = DecisionTreeRegressor(max_depth=12)
    regr_16 = DecisionTreeRegressor(max_depth=16)

    # Fit tree
    regr_2.fit(X_train, X_test)
    regr_5.fit(X_train, X_test)
    regr_8.fit(X_train, X_test)
    regr_12.fit(X_train, X_test)
    regr_16.fit(X_train, X_test)

    # Predict
    predicted_level_2 = regr_2.predict(Y_train)
    predicted_level_5 = regr_5.predict(Y_train)
    predicted_level_8 = regr_8.predict(Y_train)
    predicted_level_12 = regr_12.predict(Y_train)
    predicted_level_16 = regr_16.predict(Y_train)

    return predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16

# A support vector classifier model - TRAINING and PREDICTION
def vecto_classifier(X_train, X_test, Y_train):
    """
    Vector Classification function that returns the prediction of a list of images. With gamma 0.001
    Args:
        X_train: List of images features used for training
        X_test: List of images results used for validate the trained images.
        Y_train: List of images features predicted
    """

    #Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # We learn the digits on the first half of the digits
    classifier.fit(X_train, X_test)

    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(Y_train)

    return predicted

def CV_fold_worker(test_idx, train_idx, img_data, first_level, second_level, third_level, first_level_machine, second_level_machine, third_level_machine, return_wrapper):
    """
    Worker function for each fold in CV. Trains a model with training data, tests with
    test_idx. Places the results as (image, prediction) tuples in return wrapper
    Args:
        test_idx: List if indexes where the test_data is
        train_idx: List if indexes where the train_data is
        img_data: all of the image data
        first_level: The names of the classes, respective to model return
        return_wrapper: The list to add all results
    """
    # Create a validation set which is 10% of the training_data
    X_train, _ = util.list_split(img_data, train_idx, [0])

    Y_train, _ = util.list_split(img_data, test_idx, [0])
    Y_test_first_level, _ = util.list_split(first_level, test_idx, [0])
    Y_test_second_level, _ = util.list_split(second_level, test_idx, [0])
    Y_test_third_level, _ = util.list_split(third_level, test_idx, [0])

    X_test_first_level, _ = util.list_split(first_level, train_idx, [0])
    X_test_second_level, _ = util.list_split(second_level, train_idx, [0])
    X_test_third_level, _ = util.list_split(third_level, train_idx, [0])

    X_val_first_level = [X_test_first_level[i][1] for i in range(0,len(X_test_first_level))]
    Y_val_first_level = [Y_test_first_level[i][1] for i in range(0,len(Y_test_first_level))]

    X_val_second_level = [X_test_second_level[i][1] for i in range(0,len(X_test_second_level))]
    Y_val_second_level = [Y_test_second_level[i][1] for i in range(0,len(Y_test_second_level))]

    X_val_third_level = [X_test_third_level[i][1] for i in range(0,len(X_test_third_level))]
    Y_val_third_level = [Y_test_third_level[i][1] for i in range(0,len(Y_test_third_level))]

    list_predictions = []
    Y_train_second_level = []
    Y_train_second_level_position = []
    Y_train_third_level = []
    Y_train_third_level_position = []

    ##################################################################################################################
    # First Level of hierarchy [Mobilnet_v1]
    ##################################################################################################################
    if first_level_machine == 'nn':
        predicted = nearest_neighbour(X_train, X_val_first_level, Y_train)
    elif first_level_machine == 'dt16':
        predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val_first_level, Y_train)
        predicted = predicted_level_16
    elif first_level_machine == 'vc':
        predicted = vecto_classifier(X_train, X_val_first_level, Y_train)
    
    for position, prediction in enumerate(predicted):
        if first_level_machine == 'dt16':
            if prediction > 0.5:
                if Y_test_first_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[position][0], 1, prediction, 1, 'tf-mobilenet_v1'))
                else:
                    list_predictions.append((Y_test_first_level[position][0], 0, prediction, 1, 'tf-mobilenet_v1'))
            else:
                Y_train_second_level.append(Y_train[position])
                Y_train_second_level_position.append(position)
        else:
            if prediction == '1':
                if Y_test_first_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[position][0], 1, prediction, 1, 'tf-mobilenet_v1'))
                else:
                    list_predictions.append((Y_test_first_level[position][0], 0, prediction, 1, 'tf-mobilenet_v1'))
            else:
                Y_train_second_level.append(Y_train[position])
                Y_train_second_level_position.append(position)

    # Not necessary to go to the next level
    if len(Y_train_second_level) == 0:
        return_wrapper.append(list_predictions)
        return

    ##################################################################################################################
    # Second Level of hierarchy [Inception_v4]
    ##################################################################################################################
    #predicted = nearest_neighbour(X_train, X_val_second_level, Y_train_second_level)
    #predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val, Y_train)
    #predicted = predicted_level_16
    #predicted = vecto_classifier(X_train, X_val, Y_train)

    if second_level_machine == 'nn':
        predicted = nearest_neighbour(X_train, X_val_second_level, Y_train_second_level)
    elif second_level_machine == 'dt16':
        predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val_second_level, Y_train_second_level)
        predicted = predicted_level_16
    elif second_level_machine == 'vc':
        predicted = vecto_classifier(X_train, X_val_second_level, Y_train_second_level)

    for position, prediction in enumerate(predicted):
        if second_level_machine == 'dt16':
            if prediction > 0.5:
                if Y_test_second_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 2, prediction, 2, 'tf-inception_v4'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 0, prediction, 2, 'tf-inception_v4'))
            else:
                Y_train_third_level.append(Y_train_second_level[position])
                Y_train_third_level_position.append(Y_train_second_level_position[position])
        else:
            if prediction == '1':
                if Y_test_second_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 2, prediction, 2, 'tf-inception_v4'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 0, prediction, 2, 'tf-inception_v4'))
            else:
                Y_train_third_level.append(Y_train_second_level[position])
                Y_train_third_level_position.append(Y_train_second_level_position[position])

    if len(Y_train_third_level) == 0:
        return_wrapper.append(list_predictions)
        return

    ##################################################################################################################
    # Third Level of hierarchy [Resnet_v1_152]
    ##################################################################################################################
    #predicted = nearest_neighbour(X_train, X_val_third_level, Y_train_third_level)
    #predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val_third_level, Y_train_third_level)
    #predicted = predicted_level_16
    #predicted = vecto_classifier(X_train, X_val, Y_train)

    if third_level_machine == 'nn':
        predicted = nearest_neighbour(X_train, X_val_third_level, Y_train_third_level)
    elif third_level_machine == 'dt16':
        predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val_third_level, Y_train_third_level)
        predicted = predicted_level_16
    elif third_level_machine == 'vc':
        predicted = vecto_classifier(X_train, X_val_third_level, Y_train_third_level)

    for position, prediction in enumerate(predicted):
        if third_level_machine == 'dt16':
            if prediction > 0.5:
                if Y_test_third_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 3, 'tf-resnet_v1_152'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 3, 'tf-resnet_v1_152'))
            else:
                if Y_test_third_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 0, 'failed'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 0, 'failed'))
        else:
            if prediction == '1':
                if Y_test_third_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 3, 'tf-resnet_v1_152'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 3, 'tf-resnet_v1_152'))
            else:
                if Y_test_third_level[position][1] == '1':
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 0, 'failed'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 0, 'failed'))


    return_wrapper.append(list_predictions)

def prototype(amount_images, list_premodels):
    """
    Produce a .csv file with the fields <Image_filename, Ground truth model, predicted model>
    for every image in the train information set. We use k-fold cross validation, where k=10.
    """
    percetange_results = []

    if len(list_premodels) == 0:
        print("No premodels were selected!")
        return percetange_results
    if amount_images == 0:
        print("No images were selected!")
        return percetange_results
    
    #print("Creating training data...")
    data, first_level_data, second_level_data, third_level_data = cv_training_data(amount_images)

    for counter,(first_level_machine, second_level_machine, third_level_machine) in enumerate(list_premodels):
        # Split training data in k-fold chunks
        # Minimum needs to be 2
        k_fold = 2
        worker_threads = list()
        chunk_size = int(ceil(len(data) / float(k_fold)))
        
        # Create a new thread for each fold
        for i, (test_idx, train_idx) in enumerate(util.chunkise(range(len(data)), chunk_size)):
            return_wrapper = list()
            p = Thread(target=CV_fold_worker, args=(test_idx, train_idx, data, first_level_data, second_level_data, third_level_data, first_level_machine, second_level_machine, third_level_machine, return_wrapper))
            p.start()
            worker_threads.append((p, return_wrapper))

        # Wait for threads to finish, collect results
        all_predictions = list()
        for p, ret_val in worker_threads:
            p.join()
            all_predictions += ret_val

        predicted = []
        correct_result = []

        for p in all_predictions:
            for image, groundtruth_label, result_prediction, prediction, model_predicted in p:
                correct_result.append(groundtruth_label)
                predicted.append(prediction)
 
        percetange_results.append(compare.calculate_percentage(predicted, correct_result, [list_premodels[counter]]))
    
    return percetange_results
        
        
if __name__ == "__main__":
    cross_validation()