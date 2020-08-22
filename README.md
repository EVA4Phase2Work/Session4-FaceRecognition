# Session4-FaceRecognition
Face Recognition

This assignment is about the FaceRecognition using MT-CNN and Inception-resnet Models.

Initially the FaceImages are fed to the MT-CNN Model to extract Faces.These faces are fed to the pretrained resnet-inception model.
The Last few layers of the pretrained resnet-inception model is unfreezed while training the model for faces.

The trained model is used to predict the faces.

We used  pre-trained facenet model and then finetune it to recognize the face. 
Finetuning is greatly beneficial as we can start with the model weights already trained on a large-scale face database and then update 
some of them to reflect the new tasks we want it to perform. 
These weights already understand how to recognize faces, but the only difference is it does not know my face. 
So to have this pretrained model learn any face is much easier to train as the model weights already contain much of the 
needed information to perform the task.


## The Model PipeLine

The Model pipeline as explained above is found below:


![](https://github.com/EVA4Phase2Work/Session4-FaceRecognition/blob/master/ModelPipeline.gif)


## The DataSet

The dataset consists of 20 images of  10 Celebrities.The following celebrities are taken for predicting faces:

#### 0. Amitabh Bacchan
#### 1. Ananya Pandey
#### 2. Anil Kapoor
#### 3. Bill Gates
#### 4. Elon Musk
#### 5. Lalu Yadav
#### 6. Lionel Messi
#### 7. Micheal Jordan
#### 8. Barrack Obama
#### 9. Tom Hanks

There are 15 images of train data and 5 images of test data.

The path for dataset:

https://drive.google.com/file/d/1dy4EukqBXosU4yNU_usB5hve1_817uwz/view?usp=sharing


## Data Structure:
    The dataset has the following data structure
           data_temp
               |
               |
               test_me
                  |
                  |_ _ _train
                  |       |
                  |       |_ _ _Amitabh
                  |       |_ _ _LaLu
                  |       |_ _ _Obama
                  |       |_ _ _
                  |
                  |_ _ _val
                          |      
                          |_ _ _Amitabh
                          |_ _ _LaLu
                          |_ _ _Obama
                          |_ _ _
                   
               
## HTML File for Prediction:
    
    http://evadebs1.s3-website.ap-south-1.amazonaws.com

## Code Structure:

#### face_latest_training.ipynb (https://github.com/EVA4Phase2Work/Session4-FaceRecognition/blob/master/face_latest_training.ipynb)- Jupyter Notebook for training the model
#### mtcnn_processing.py (https://github.com/EVA4Phase2Work/Session4-FaceRecognition/blob/master/mtcnn_processing.py) - Python file for porcessing images to get the face using MTCNN model
##### handler.py - Serverless Handler Python file
###### serverless.yml - Serverless deployment yml file


    
 ## Accuracy:
 
 We got best validation accuracy of 88%
 
train Loss: 2.2943 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8600
Training complete in 1m 46s
Best val Acc: 0.880000


Full Logs:
Epoch 0/99
----------
train Loss: 2.3030 Acc: 0.1000
val Loss: 2.3011 Acc: 0.2800
Epoch 1/99
----------
train Loss: 2.2997 Acc: 0.3438
val Loss: 2.2975 Acc: 0.6200
Epoch 2/99
----------
train Loss: 2.2964 Acc: 0.6813
val Loss: 2.2954 Acc: 0.7400
Epoch 3/99
----------
train Loss: 2.2954 Acc: 0.7688
val Loss: 2.2950 Acc: 0.8200
Epoch 4/99
----------
train Loss: 2.2948 Acc: 0.7875
val Loss: 2.2945 Acc: 0.8400
Epoch 5/99
----------
train Loss: 2.2947 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8200
Epoch 6/99
----------
train Loss: 2.2946 Acc: 0.7812
val Loss: 2.2946 Acc: 0.8600
Epoch 7/99
----------
train Loss: 2.2944 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8400
Epoch 8/99
----------
train Loss: 2.2948 Acc: 0.7750
val Loss: 2.2945 Acc: 0.8400
Epoch 9/99
----------
train Loss: 2.2945 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8200
Epoch 10/99
----------
train Loss: 2.2943 Acc: 0.8438
val Loss: 2.2944 Acc: 0.8400
Epoch 11/99
----------
train Loss: 2.2944 Acc: 0.8375
val Loss: 2.2945 Acc: 0.8400
Epoch 12/99
----------
train Loss: 2.2947 Acc: 0.7938
val Loss: 2.2944 Acc: 0.8400
Epoch 13/99
----------
train Loss: 2.2945 Acc: 0.8188
val Loss: 2.2944 Acc: 0.8400
Epoch 14/99
----------
train Loss: 2.2945 Acc: 0.8188
val Loss: 2.2945 Acc: 0.8200
Epoch 15/99
----------
train Loss: 2.2942 Acc: 0.8188
val Loss: 2.2944 Acc: 0.8400
Epoch 16/99
----------
train Loss: 2.2946 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8400
Epoch 17/99
----------
train Loss: 2.2945 Acc: 0.8375
val Loss: 2.2944 Acc: 0.8200
Epoch 18/99
----------
train Loss: 2.2945 Acc: 0.8125
val Loss: 2.2944 Acc: 0.8200
Epoch 19/99
----------
train Loss: 2.2946 Acc: 0.8375
val Loss: 2.2945 Acc: 0.8400
Epoch 20/99
----------
train Loss: 2.2946 Acc: 0.7750
val Loss: 2.2945 Acc: 0.8200
Epoch 21/99
----------
train Loss: 2.2945 Acc: 0.8250
val Loss: 2.2945 Acc: 0.8400
Epoch 22/99
----------
train Loss: 2.2945 Acc: 0.8188
val Loss: 2.2944 Acc: 0.8600
Epoch 23/99
----------
train Loss: 2.2943 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8400
Epoch 24/99
----------
train Loss: 2.2944 Acc: 0.8250
val Loss: 2.2944 Acc: 0.8400
Epoch 25/99
----------
train Loss: 2.2947 Acc: 0.8188
val Loss: 2.2945 Acc: 0.8400
Epoch 26/99
----------
train Loss: 2.2942 Acc: 0.8563
val Loss: 2.2946 Acc: 0.8200
Epoch 27/99
----------
train Loss: 2.2945 Acc: 0.8000
val Loss: 2.2945 Acc: 0.8400
Epoch 28/99
----------
train Loss: 2.2946 Acc: 0.8000
val Loss: 2.2945 Acc: 0.8200
Epoch 29/99
----------
train Loss: 2.2944 Acc: 0.8063
val Loss: 2.2945 Acc: 0.8200
Epoch 30/99
----------
train Loss: 2.2946 Acc: 0.7625
val Loss: 2.2944 Acc: 0.8200
Epoch 31/99
----------
train Loss: 2.2944 Acc: 0.8375
val Loss: 2.2944 Acc: 0.8400
Epoch 32/99
----------
train Loss: 2.2947 Acc: 0.7875
val Loss: 2.2945 Acc: 0.8000
Epoch 33/99
----------
train Loss: 2.2943 Acc: 0.8625
val Loss: 2.2945 Acc: 0.8400
Epoch 34/99
----------
train Loss: 2.2944 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8400
Epoch 35/99
----------
train Loss: 2.2946 Acc: 0.8063
val Loss: 2.2944 Acc: 0.8400
Epoch 36/99
----------
train Loss: 2.2945 Acc: 0.8125
val Loss: 2.2944 Acc: 0.8600
Epoch 37/99
----------
train Loss: 2.2945 Acc: 0.7938
val Loss: 2.2945 Acc: 0.8400
Epoch 38/99
----------
train Loss: 2.2943 Acc: 0.8313
val Loss: 2.2944 Acc: 0.8400
Epoch 39/99
----------
train Loss: 2.2945 Acc: 0.8500
val Loss: 2.2945 Acc: 0.8400
Epoch 40/99
----------
train Loss: 2.2945 Acc: 0.7812
val Loss: 2.2945 Acc: 0.8600
Epoch 41/99
----------
train Loss: 2.2942 Acc: 0.8438
val Loss: 2.2945 Acc: 0.8400
Epoch 42/99
----------
train Loss: 2.2945 Acc: 0.7875
val Loss: 2.2945 Acc: 0.8400
Epoch 43/99
----------
train Loss: 2.2948 Acc: 0.7625
val Loss: 2.2944 Acc: 0.8200
Epoch 44/99
----------
train Loss: 2.2947 Acc: 0.8000
val Loss: 2.2944 Acc: 0.8400
Epoch 45/99
----------
train Loss: 2.2945 Acc: 0.8250
val Loss: 2.2945 Acc: 0.8200
Epoch 46/99
----------
train Loss: 2.2949 Acc: 0.7875
val Loss: 2.2944 Acc: 0.8400
Epoch 47/99
----------
train Loss: 2.2947 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8400
Epoch 48/99
----------
train Loss: 2.2944 Acc: 0.8000
val Loss: 2.2945 Acc: 0.8000
Epoch 49/99
----------
train Loss: 2.2945 Acc: 0.8000
val Loss: 2.2945 Acc: 0.8200
Epoch 50/99
----------
train Loss: 2.2943 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8200
Epoch 51/99
----------
train Loss: 2.2948 Acc: 0.7938
val Loss: 2.2944 Acc: 0.8000
Epoch 52/99
----------
train Loss: 2.2945 Acc: 0.8438
val Loss: 2.2944 Acc: 0.8400
Epoch 53/99
----------
train Loss: 2.2946 Acc: 0.8375
val Loss: 2.2944 Acc: 0.8600
Epoch 54/99
----------
train Loss: 2.2942 Acc: 0.8688
val Loss: 2.2945 Acc: 0.8200
Epoch 55/99
----------
train Loss: 2.2945 Acc: 0.8250
val Loss: 2.2944 Acc: 0.8400
Epoch 56/99
----------
train Loss: 2.2945 Acc: 0.8000
val Loss: 2.2944 Acc: 0.8400
Epoch 57/99
----------
train Loss: 2.2945 Acc: 0.7938
val Loss: 2.2945 Acc: 0.8400
Epoch 58/99
----------
train Loss: 2.2946 Acc: 0.8000
val Loss: 2.2945 Acc: 0.8200
Epoch 59/99
----------
train Loss: 2.2946 Acc: 0.7875
val Loss: 2.2945 Acc: 0.8400
Epoch 60/99
----------
train Loss: 2.2946 Acc: 0.8250
val Loss: 2.2945 Acc: 0.8400
Epoch 61/99
----------
train Loss: 2.2943 Acc: 0.8188
val Loss: 2.2945 Acc: 0.8400
Epoch 62/99
----------
train Loss: 2.2944 Acc: 0.7750
val Loss: 2.2944 Acc: 0.8400
Epoch 63/99
----------
train Loss: 2.2945 Acc: 0.8125
val Loss: 2.2944 Acc: 0.8600
Epoch 64/99
----------
train Loss: 2.2943 Acc: 0.8500
val Loss: 2.2944 Acc: 0.8400
Epoch 65/99
----------
train Loss: 2.2946 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8400
Epoch 66/99
----------
train Loss: 2.2943 Acc: 0.8250
val Loss: 2.2944 Acc: 0.8400
Epoch 67/99
----------
train Loss: 2.2944 Acc: 0.8063
val Loss: 2.2945 Acc: 0.8200
Epoch 68/99
----------
train Loss: 2.2946 Acc: 0.8125
val Loss: 2.2944 Acc: 0.8400
Epoch 69/99
----------
train Loss: 2.2945 Acc: 0.8188
val Loss: 2.2944 Acc: 0.8400
Epoch 70/99
----------
train Loss: 2.2943 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8200
Epoch 71/99
----------
train Loss: 2.2944 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8200
Epoch 72/99
----------
train Loss: 2.2944 Acc: 0.8000
val Loss: 2.2945 Acc: 0.8200
Epoch 73/99
----------
train Loss: 2.2943 Acc: 0.8375
val Loss: 2.2944 Acc: 0.8400
Epoch 74/99
----------
train Loss: 2.2946 Acc: 0.8063
val Loss: 2.2944 Acc: 0.8200
Epoch 75/99
----------
train Loss: 2.2946 Acc: 0.7938
val Loss: 2.2945 Acc: 0.8200
Epoch 76/99
----------
train Loss: 2.2944 Acc: 0.8063
val Loss: 2.2945 Acc: 0.8400
Epoch 77/99
----------
train Loss: 2.2943 Acc: 0.8125
val Loss: 2.2944 Acc: 0.8400
Epoch 78/99
----------
train Loss: 2.2946 Acc: 0.7625
val Loss: 2.2945 Acc: 0.8400
Epoch 79/99
----------
train Loss: 2.2945 Acc: 0.8000
val Loss: 2.2944 Acc: 0.8600
Epoch 80/99
----------
train Loss: 2.2943 Acc: 0.7938
val Loss: 2.2945 Acc: 0.8400
Epoch 81/99
----------
train Loss: 2.2945 Acc: 0.8188
val Loss: 2.2945 Acc: 0.8400
Epoch 82/99
----------
train Loss: 2.2946 Acc: 0.7750
val Loss: 2.2945 Acc: 0.8400
Epoch 83/99
----------
train Loss: 2.2947 Acc: 0.8063
val Loss: 2.2944 Acc: 0.8800
Epoch 84/99
----------
train Loss: 2.2943 Acc: 0.8313
val Loss: 2.2944 Acc: 0.8600
Epoch 85/99
----------
train Loss: 2.2945 Acc: 0.8313
val Loss: 2.2944 Acc: 0.8600
Epoch 86/99
----------
train Loss: 2.2946 Acc: 0.8188
val Loss: 2.2944 Acc: 0.8400
Epoch 87/99
----------
train Loss: 2.2945 Acc: 0.8063
val Loss: 2.2944 Acc: 0.8400
Epoch 88/99
----------
train Loss: 2.2947 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8400
Epoch 89/99
----------
train Loss: 2.2946 Acc: 0.8063
val Loss: 2.2945 Acc: 0.8000
Epoch 90/99
----------
train Loss: 2.2944 Acc: 0.8250
val Loss: 2.2945 Acc: 0.8200
Epoch 91/99
----------
train Loss: 2.2945 Acc: 0.8125
val Loss: 2.2944 Acc: 0.8600
Epoch 92/99
----------
train Loss: 2.2945 Acc: 0.8375
val Loss: 2.2945 Acc: 0.8400
Epoch 93/99
----------
train Loss: 2.2943 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8200
Epoch 94/99
----------
train Loss: 2.2943 Acc: 0.8188
val Loss: 2.2944 Acc: 0.8600
Epoch 95/99
----------
train Loss: 2.2946 Acc: 0.8125
val Loss: 2.2945 Acc: 0.8400
Epoch 96/99
----------
train Loss: 2.2944 Acc: 0.8188
val Loss: 2.2944 Acc: 0.8400
Epoch 97/99
----------
train Loss: 2.2944 Acc: 0.8188
val Loss: 2.2945 Acc: 0.8200
Epoch 98/99
----------
train Loss: 2.2945 Acc: 0.8063
val Loss: 2.2945 Acc: 0.8400
Epoch 99/99
----------
train Loss: 2.2943 Acc: 0.8313
val Loss: 2.2945 Acc: 0.8600
Training complete in 1m 46s
Best val Acc: 0.880000

## Loss Plot

![Loss Plot](/loss_plot.png)
    
