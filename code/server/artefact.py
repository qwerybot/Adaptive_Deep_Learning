from __future__ import print_function

import Pyro4
import analysis
import sys

import numpy as np

from image_generator import ImageGenerator
import ipywidgets as widgets

# The image numbers of our motivation images
img_num_motivation = [97, 36861, 32781]

def combine_model_results(master, appendix):
    """
    Appends the model results from appendix onto master, returns the new tuple

    Args:
        master (tuple): The place to accumulate all results
        appendix (tuple): The new results to add

    Returns:
        tuple: The result of adding appendix to master
    """
    if len(master) == 0:
        return appendix

    ret_val = list()
    for i in range(len(master)):
        ret_val.append(master[i] + appendix[i])

    return tuple(ret_val)


def find_dnn_daemon():
    """
    Return a reference to the DNN daemon
    """
    # HOST_IP should match the IP address used when you set up the nameserver
    # HOST_PORT is found by looking at the port your nameserver is using once
    # you start it.
    HOST_IP = '148.88.227.201'
    HOST_PORT = 9090

    # Return the dnn_daemon if found
    with Pyro4.locateNS(host=HOST_IP, port=HOST_PORT) as ns:
        uri = ns.lookup("artefact.dnn_daemon")
        dnn_daemon = Pyro4.Proxy(uri)

    #model_names = dnn_daemon.available_models()
    #print("There are", len(model_names), "models available.")

    return dnn_daemon


def convert_prediction(prediction_vals, prediction_classes, class_names):
    """
    Return a list of tuples in order of predition confidence
    (class_string, %_confidence)
    """
    sorted_results = [i[0] for i in sorted(enumerate(-np.array(prediction_vals)),
                                           key=lambda x:x[1])]

    ret_val = list()
    for i in range(len(prediction_vals)):
        index = sorted_results[i]
        ret_val.append((class_names[prediction_classes[index]],
                        100*prediction_vals[index]))
    return ret_val


def model_available(dnn_daemon, model):
    """
    Return True if model is availabel on daemon, else False
    """
    available_models = dnn_daemon.available_models()

    if model in available_models:
        return True
    return False

def model_available_list():
    """
    Return True if model is availabel on daemon, else False
    """
    
    dnn_daemon = find_dnn_daemon()
    available_models = []
    for model in dnn_daemon.available_models():
        available_models.append(model)

    return available_models

def img_filename_motivation():
    """
    Returns the number of the images from the motivation. It can be change if we want to modify them
    """
    # The image numbers of our motivation images
    img_nums = img_num_motivation

    # Initialise an image generator
    img_dir = 'images/val/images/'
    img_class_map_file = 'images/val/val.txt'
    class_file = 'images/val/synset_words.txt'
    image_generator = ImageGenerator(img_dir, img_class_map_file, class_file)
    
    # Get a path and label for each of our images
    img_filenames = list()
    
    for img in img_nums:
        filename=image_generator.get_image_filename(img)
        img_filenames.append(filename)
    
    return img_filenames

def images_motivation(dnn_daemon):
    """
    Plots the images from the motivation section
    """
   # The image numbers of our motivation images
    img_nums = img_num_motivation

    # Initialise an image generator
    img_dir = 'images/val/images/'
    img_class_map_file = 'images/val/val.txt'
    class_file = 'images/val/synset_words.txt'
    image_generator = ImageGenerator(img_dir, img_class_map_file, class_file)

    # Get a path and label for each of our images
    img_paths = list()
    img_labels = list()
    for img in img_nums:
        path, label = image_generator.get_image_data(img)
        img_paths.append(path)
        img_labels.append(label)

    return img_paths
            
# img_nums = List of image numbers
# model_names = List of model names
def inference(dnn_daemon, img_nums, model_names, n_img_to_infer=5):
    """
    Allows to do inference using a list of images and models.
    """
    
    # Check all models are available
    for model in model_names:
        if not model_available(dnn_daemon, model):
            print("ERROR: Model", model, "not availale. Exiting.")
            sys.exit()

    # Initialise an image generator
    img_dir = 'images/val/images/'
    img_class_map_file = 'images/val/val.txt'
    class_file = 'images/val/synset_words.txt'
    image_generator = ImageGenerator(img_dir, img_class_map_file, class_file)

    # Get a path and label for each of our images
    img_paths = list()
    img_labels = list()
    for img in img_nums:
        path, label = image_generator.get_image_data(img)
        img_paths.append(path)
        img_labels.append(label)

    # Infer each image with each model, get the results
    # results: dict, mapping (img_num, model) to (inference_times, prediciton)
    print("Running inference on Jetson...")
    img_model_to_results = dict()
    img_model_to_img = []
    img_model_to_models = []
    img_model_to_inference = []
    img_model_to_prediction = []
    img_model_to_results = []
    
    if(len(model_names) is 0):
        print("Any model was selected!!!")
        return img_model_to_img, img_model_to_models, img_model_to_results
    
    percentage_of_each_execution=100/(len(img_paths)*len(model_names))
    counter=0
    first_time=0
    for img_num, path in enumerate(img_paths):
        img_model_to_inference = []
        img_model_to_img.append("Image " + `img_num`)
        for model in model_names:
            results = dnn_daemon.inference(model, [path], 1)
            inference_time, prediction_vals, prediction_classes = results
            inference_time = inference_time[0]
            ordered_prediction = convert_prediction(prediction_vals[0],
                                                    prediction_classes[0],
                                                    image_generator.class_names)
            #img_model_to_results[(img_num, model)] = (inference_time, ordered_prediction)
            
            if first_time is 0:
                img_model_to_models.append(model)
                
            img_model_to_inference.append(inference_time)
            #img_model_to_prediction.append(ordered_prediction)
            counter=counter+1
            print(percentage_of_each_execution*counter, "%\n")
            
        img_model_to_results.append(img_model_to_inference)
        first_time=first_time+1
        
    #print("All the images has been inferenced! ")
    return img_model_to_img, img_model_to_models, img_model_to_results
   

def motivation_example(dnn_daemon, model_names):
    """
    Perform the images from the motivation in our paper using the models that the reviewer wants.
    """
    # The image numbers of our motivation images
    img_nums = img_num_motivation
    img_model_to_img, img_model_to_models, img_model_to_results = inference(dnn_daemon, img_nums, model_names)
    
    return img_model_to_img, img_model_to_models, img_model_to_results


def main():
    # Connect to the remote daemon for managing inference
    dnn_daemon = find_dnn_daemon()

    # First, create the graphs for motivation
    print("Recreating motivation...")
    motivation(dnn_daemon)

    # test_inference(dnn_daemon)
    print("Hello World!")


if __name__ == "__main__":
    main()
