from __future__ import print_function
import time

import numpy as np


def top_n_correct(predictions, class_numbers, correct_label, label_list, n):
    """Return True if the highest confidence is the correct label"""
    sorted_results = [i[0] for i in sorted(enumerate(-predictions), key=lambda x:x[1])]


    # Check the top n labels
    for i in range(n):
        index = sorted_results[i]
        try:
            predicted_label = label_list[class_numbers[index]]
        except IndexError:
            # Higher class number than available
            predicted_label = label_list[-1]

        if predicted_label == correct_label:
            return True

    return False


def print_probability_results(predictions, class_numbers, correct_label, label_list, n=5):

    sorted_results = [i[0] for i in sorted(enumerate(-predictions), key=lambda x:x[1])]
    
    print("correct label:", correct_label)
    for i in range(n):
        index = sorted_results[i]
        print('Probability %0.2f%% => [%s]' % (100*predictions[index], label_list[class_numbers[index]]))


def top_n_labels(predictions, class_numbers, label_list, n):
    """Return top n labels from predictions"""
    sorted_results = [i[0] for i in sorted(enumerate(-predictions), key=lambda x:x[1])]

    top_n_labels = list()
    for i in range(n):
        index = sorted_results[i]
        try:
            predicted_label = label_list[class_numbers[index]]
        except IndexError:
            # Higher class number than avialable
            predicted_label = label_list[-1]

        top_n_labels.append(predicted_label)

    return top_n_labels


def top_n_prediction(results, label_list, n):
    """Return a list of (label, confidence) pairs of the top n predictions"""
    sorted_results = [i[0] for i in sorted(enumerate(-results), key=lambda x:x[1])]

    ret_val = list()
    for i in range(n):
        index = sorted_results[i]
        confidence = 100*results[index]
        label = label_list[index]
        ret_val.append((label, confidence))

    return ret_val


def prediction_accuracy(results, correct_label, n=2):
    """Returns the confidence of the correct label"""
    for label, confidence in top_n_prediction(results, n):
        if label == correct_label:
            return confidence

    return 0


def current_milli_time():
    return int(round(time.time() * 1000))


def time_function(function, *args):

    start_time = current_milli_time()
    return_val = function(*args)
    end_time = current_milli_time()

    return (end_time - start_time, return_val)


def reduce_prediction_vals(prediction_array, n):
    """
    Reduces the size of the prediction array to the top n vals

    Args:
        prediction_array (1D numpy array): the prediction values from a model
        n (int): the number of values to return

    Return:
        1D numpy array: the top n predictions
    """
    sorted_results = [i[0] for i in sorted(enumerate(-prediction_array), key=lambda x:x[1])]

    pred_vals = list()
    class_numbers = list()
    for i in range(n):
        index = sorted_results[i]
        pred_vals.append(prediction_array[index])
        class_numbers.append(index)

    return (class_numbers, np.array(pred_vals))


def generate_per_image_data(model_data):
    """
    Converts the dictionary of model data and generates data on a per image basis

    Args:
        model_data (dict): A dictionary of the model values

    Yields:
        list: A list of tuples containing all the model data for an image
    """
    n_images = len(model_data.values()[0][0])

    for i in range(n_images):
        image_data = dict()
        for model in model_data.keys():
            inference_val_list, prediction_val_list, class_num_list = model_data[model]
            image_data[model.full_name] = (inference_val_list[i], prediction_val_list[i], 
                                           class_num_list[i])
        yield image_data


def top_n_best_model(image_data, correct_label, label_list, n):
    """
    Calculates which model from image data was the best predictor in the n top predictions
    """

    # Check if model matches for top n
    correct_models = list()
    for model in image_data.keys():
        _, prediction, classes = image_data[model]

        if top_n_correct(prediction, classes, correct_label, label_list, n):
            correct_models.append(model)

    if correct_models == list():
        return 'failed'

    # Get the model with the smallest inference time
    inference_times = list()
    for model in correct_models:
        inference, _, _= image_data[model]
        inference_times.append(inference)

    return correct_models[inference_times.index(min(inference_times))]


def list_chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Yields:
        a list of length n
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]
