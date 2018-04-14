from __future__ import print_function
from abc import ABCMeta, abstractmethod
from threading import Thread


class Model():
    """
    An Abstract class for calling each model
    Each framework is slightly different, so this allows them all to be called in the same way
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def time_model(self, image_path, iterations, return_wrapper):
        """
        Sets up model and runs inference iterations times

        Returns:
            int: The average time of inference on the model in ms
            1D numpy array: The output of the model prediction
            list: The classes the predictions belong to
        """
        pass

    def time_model_thread(self, image_path, iterations):
        """
        A single thread wrapper for the time_model function
        This is used to give each framework its own context of the GPU

        Args:
            image_path (string): Path to image for inference
            iterations (int): Number of times to repeat inference

        Returns:
            int: The average time of inference on the model in ms
            1D numpy array: The output of the model prediction
            list: The classes the predictions belong to
        """
        return_wrapper = list()

        p = Thread(target=self.time_model, args=(image_path, iterations, return_wrapper))
        p.start()
        p.join()

        return return_wrapper[0]
