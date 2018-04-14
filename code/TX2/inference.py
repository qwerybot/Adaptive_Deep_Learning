from __future__ import print_function

from copy import deepcopy
from nets import nets_factory
from model import Model
from preprocessing import preprocessing_factory

import tensorflow as tf
import Pyro4
import analysis
import os

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_classes_map = {
    'inception_v1': 1001,
    'inception_v2': 1001,
    'inception_v3': 1001,
    'inception_v4': 1001,
    'inception_resnet_v2': 1001,
    'resnet_v1_50':  1000,
    'resnet_v1_101': 1000,
    'resnet_v1_152': 1000,
    'resnet_v2_50': 1001,
    'resnet_v2_101': 1001,
    'resnet_v2_152': 1001,
    'vgg_16': 1000,
    'vgg_19': 1000,
    'mobilenet_v1': 1001,
    'mobilenet_v1_075': 1001,
    'mobilenet_v1_050': 1001,
    'mobilenet_v1_025': 1001
    }


class DNN_Model(Model):

    def __init__(self, framework, model_name, weights_path):
        self.framework = framework
        self.model_name = model_name
        self.full_name = framework + '-' + model_name
        self._weights_path = weights_path
        self.skip_first_inference = True

    def inference(self, sess, image, probabilities):
        """
        Runs image though the model defined through init_function
        """
        np_image, probabilities, = sess.run([image, probabilities])

        return probabilities

    def _get_image(self, image_path):
        with open(image_path, 'r') as f:
            data = f.read()
        image = tf.image.decode_jpeg(data, channels=3)
        return image

    def _preprocess_image(self, image, network_fn):
        # Special case if resnet_v2 model
        if self.model_name[:9] == 'resnet_v2':
            image_size = 299
        else:
            image_size = network_fn.default_image_size

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(self.model_name,
                                                                         is_training=False)
        processed_image = image_preprocessing_fn(image, image_size, image_size)
        return tf.expand_dims(processed_image, 0)

    def time_model(self, image_path_list, iterations, return_wrapper):
        """
        Sets up model and runs inference iterations times

        Returns:
            int: The average time of inference on the model in ms
            1D numpy array: The output of the model prediction
            list: The classes the predictions belong to
        """

        ####################
        # Select the model #
        ####################
        if self.model_name not in num_classes_map:
            raise ValueError('Name of network unknown %s' % self.model_name)

        ##############################
        # First inference is anomoly #
        # Throw away data            #
        ##############################
        if self.skip_first_inference:
            self.skip_first_inference = False
            print('Skipping first inference...')
            self.time_model([image_path_list[0]], iterations, deepcopy(return_wrapper))
            print('Done!\n')

        ############################
        # Select the correct model #
        ############################
        network_fn = nets_factory.get_network_fn(self.model_name,
                                                 num_classes=num_classes_map[self.model_name],
                                                 is_training=False)
        #################################
        # Run in batches to save memory #
        #################################
        # Set batch to all images - ignore batch code for now...
        batch_size = 1 #len(image_path_list)
        for batch_num, image_batch in enumerate(analysis.list_chunks(image_path_list, batch_size)):
            tf.reset_default_graph()

            #####################
            # Preprocess images #
            #####################
            inference_imgs = list()
            inference_probs = list()
            for img_num, image_path in enumerate(image_batch):
                image_path = str(image_path)
                print("Preprocessing image: " + str((batch_num * batch_size) + img_num + 1) +
                      " of " + str(len(image_path_list)))
                with open(image_path, 'r') as f:
                    image = tf.image.decode_jpeg(f.read(), channels=3)
                    inference_imgs.append(image)

                    processed_image = self._preprocess_image(image, network_fn)
                    logits, _ = network_fn(processed_image)
                    inference_probs.append(tf.nn.softmax(logits))

            ######################
            # Get the checkpoint #
            ######################
            init_fn = slim.assign_from_checkpoint_fn(self._weights_path,
                                                     slim.get_variables_to_restore())

            #################
            # Run the model #
            #################
            with tf.Session() as sess:
                init_fn(sess)
                prediction_list = list()
                class_list = list()
                average_times = list()

                for img_num, (image, probabilities) in enumerate(zip(inference_imgs, inference_probs)):

                    all_times = list()
                    for i in range(iterations):
                        time_taken, prediction = analysis.time_function(self.inference, sess,
                                                                        image, probabilities)
                        all_times.append(time_taken)
                    average_time = sum(all_times)/float(iterations)

                    print("Image: " + str((batch_num * batch_size) + img_num + 1) +
                          " of " + str(len(image_path_list)) + "\tAverage Time:", average_time)

                    class_offset = num_classes_map[self.model_name] - 1000
                    classes, prediction = analysis.reduce_prediction_vals(prediction[0, class_offset:],
                                                                          20)
                    prediction_list.append(prediction.tolist())
                    class_list.append(classes)
                    average_times.append(average_time)

            return_wrapper.append((average_times, prediction_list, class_list))


@Pyro4.expose
class DNN_Daemon():
    """
    Class designed to expose the DNNs to the daemon
    """
    def __init__(self, num_models=None):
        self.dnn_models = self._initialise_DNN_models(num_models)
        self.inference_count = 0

    def _initialise_DNN_models(self, num_models=None):
        """
        Return model names and DNN_Model Objects
        """
        all_models = list()
        tf_weights = self._tf_checkpoint_map()
        for model_num, (model_name, weights_path) in enumerate(tf_weights.iteritems()):
            if model_num == num_models:
                break
            all_models.append(DNN_Model('tf', model_name, weights_path))

        return all_models

    def _tf_checkpoint_map(self):
        """
        Get the map from model names to paths for the tf model checkpoints

        Returns:
            dict: Keys are model names, values are paths to model description
        """
        model_data_prefix = 'model_data/tensorflow/checkpoints/'

        checkpoint_map = {
            'inception_v1':        os.path.join(model_data_prefix, 'inception_v1/inception_v1.ckpt'),
            'inception_v2':        os.path.join(model_data_prefix, 'inception_v2/inception_v2.ckpt'),
            'inception_v4':        os.path.join(model_data_prefix, 'inception_v4/inception_v4.ckpt'),
            'resnet_v1_50':        os.path.join(model_data_prefix, 'resnet_v1_50/resnet_v1_50.ckpt'),
            'resnet_v1_101':       os.path.join(model_data_prefix, 'resnet_v1_101/resnet_v1_101.ckpt'),
            'resnet_v1_152':       os.path.join(model_data_prefix, 'resnet_v1_152/resnet_v1_152.ckpt'),
            'resnet_v2_50':        os.path.join(model_data_prefix, 'resnet_v2_50/resnet_v2_50.ckpt'),
            'resnet_v2_101':       os.path.join(model_data_prefix, 'resnet_v2_101/resnet_v2_101.ckpt'),
            'resnet_v2_152':       os.path.join(model_data_prefix, 'resnet_v2_152/resnet_v2_152.ckpt'),
            'mobilenet_v1':        os.path.join(model_data_prefix, 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'),
            'mobilenet_v1_075':    os.path.join(model_data_prefix, 'mobilenet_v1_0.75_192/mobilenet_v1_0.75_192.ckpt'),
            'mobilenet_v1_050':    os.path.join(model_data_prefix, 'mobilenet_v1_0.50_160/mobilenet_v1_0.50_160.ckpt'),
            'mobilenet_v1_025':    os.path.join(model_data_prefix, 'mobilenet_v1_0.25_128/mobilenet_v1_0.25_128.ckpt')
            }

        return checkpoint_map

    def _get_model(self, model_name):
        """
        Return the model object matching model_name
        """
        print(model_name)
        for dnn_model in self.dnn_models:
            if dnn_model.model_name == model_name:
                return dnn_model

        print("COULDN'T FIND MODEL, WE GOT A PROBLEM!!")

    def inference(self, model_name, image_path_list, iterations):
        """
        Pass the inference job to the specified model, return the results
        """
        self.inference_count += 1
        print(10*"-", "Serving Inference Request",
              self.inference_count, 10*"-")
        model = self._get_model(model_name)
        results = model.time_model_thread(image_path_list, iterations)
        print(10*"-", "Inference Request", self.inference_count,
              "complete", 10*"-", "\n")
        return results

    def available_models(self):
        """
        Return the names of models which are available
        """
        model_names = list()
        for model in self.dnn_models:
            model_names.append(model.model_name)
        return model_names


def main():

    dnn_daemon = DNN_Daemon()

    # Set this to the IP address of the thing hosting inference
    HOST_IP = '148.88.227.201'
    # Make sure you have also run the command 'pyro4-ns -n HOST_IP'
    # before starting up this, otherwise there will be no name server to 
    # connect to!
    # An example can be found in 'start_name_server' which can be found in
    # the same directory as this file

    # Set up daemon to serve inference requests for each model
    with Pyro4.Daemon(host=HOST_IP) as daemon:
        dnn_uri = daemon.register(dnn_daemon)
        with Pyro4.locateNS() as ns:
            ns.register("artefact.dnn_daemon", dnn_uri)
        print("DNN Daemon Available.")
        daemon.requestLoop()


if __name__ == '__main__':
    main()
