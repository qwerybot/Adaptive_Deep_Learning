# Adaptive Deep Learning Model Selection On Embedded Systems

Ben Taylor, Vicent Sanz Marco, Willy Wolff, Yehia Elkhatib, Zheng Wang

** Abstract **
> The recent ground-breaking advances in deep learning networks (DNNs) make them attractive for embedded systems.
> However, it can take a long time for DNNs to make an inference on resource-limited embedded devices. 
> Offloading the computation into the cloud is often infeasible due to privacy concerns, high latency, or the lack of connectivity. 
> As such, there is a critical need to find a way to effectively execute the DNN models locally on the devices.
>
> This paper presents an adaptive scheme to determine which DNN model to use for a given input, by considering the desired 
> accuracy and inference time. Our approach employs machine learning to develop a predictive model to quickly select a pre-trained 
> DNN to use for a given input and the optimization constraint. We achieve this by first training off-line a predictive model, and 
> then use the learnt model to select a DNN model to use for new, unseen inputs. We apply our approach to the image classification 
> task and evaluate it on a Jetson TX2 embedded deep learning platform using the ImageNet ILSVRC 2012 validation dataset. We 
> consider a range of influential DNN models. Experimental results show that our approach achieves a 7.52% improvement in 
> inference accuracy, and a 1.8x reduction in inference time over the most-capable, single DNN model


## Getting Started

This was written for an artefact evaluation which we had pre-deployed onto our own pre-configured server, we have described below how
to set up this system in your own environment for your own use.

### Requirements
* [Python2](https://www.python.org/downloads/)
* [Tensorflow 1.3](https://www.tensorflow.org/versions/r1.3/)
* [numpy](https://www.scipy.org/scipylib/download.html)
* [Pyro4](https://pythonhosted.org/Pyro4)
* [MySQLdb](http://mysqlclient.readthedocs.io)
* [scikit-learn](http://scikit-learn.org/stable/install.html)

### Optional Requirements
* [Jupyter Notebook](http://jupyter.org/install.html)

#### External Data 
Our artefact requires two lots of external data which we are unable to store in this repostory due to their size:

* [ILVRSC 2012 validation image set](http://www.image-net.org/download-images) - The 50k images which were used as a validation set in the 2012 ILVRSC imagenet challenge. 
    These should be stored in the folder `images\val\images`.
* DNN checkpoints - The checkpoints for the pretrained models provided by the tf-slim library. All these checkpoints can be found at:
    [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim). These checkpoints should be stored in the model_data folder at 
    the root of this artefact. e.g. The Inception_v4 checkpoints should be stored at `model_data\tensorflow\checkpoints\inception_v4`

## Deployment
In our paper we evaluate our premodel on a JetsonTX2 embedded system, and we train our models on a powerful server.
The majority of the work is carried out on the server, with the actual image inference being performed on a Jetson TX2 embedded system.
This approach kept the results accurate while making model training and computation as fast as possible for the artefact evaluation.
Below we describe how to set up the embedded system first, then how to connect the server for remote inference.

### Jetson TX2 setup
To perform remote inference on another system you are going to need to obtain the ip address through methods such as `ifconfig`. From
now on we will refer to your ip address as `your_ip`. We will use that in place of anywhere you need to include your ip address.

1. Traverse to `code/TX2`.

    `cd code\TX2`

2. Set up the nameserver

    `pyro4-ns -n your_ip`

3. Take note of the port which was output by the previous step, we refer to this as `your_port` from now on. e.g.

    `URI = PYRO:Pyro.NameServer@127.0.0.1:9090`

    means `your_port` equals 9090

4. Look in `inference.py`. At the top if this file is a variable `HOST_IP` set this to `your_ip`. e.g.

    `HOST_IP = 127.0.0.1`

5. Run `inference.py`. This should connect to the name server and set up a service you can connect to for performing remote inference.

    `python inference.py`


### Jupyter Notebook setup

Inside the folder `code\server\` there is a file called `Artefact.ipynb` that is used by Jupyter Notebook to open a web application allowing to perform the experiments showed in the paper.


1. First install Jupyter Notebook.


2. Go to the folder `code\server\` using your Terminal (Mac/Linux) or Command Prompt (Windows).


3. Type the following command: 

    `jupyter notebook`


4. This will print some information about the notebook server in your terminal, including the URL of the web application (by default, http://localhost:8888).




