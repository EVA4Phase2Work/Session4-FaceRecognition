# Session4-FaceRecognition
Face Recognition

This assignment is about the FaceRecognition using MT-CNN and Inception-resnet Models.

Initially the FaceImages are fed to the MT-CNN Model to extract Faces.These faces are fed to the pretrained resnet-inception model.
The Last few layers of the pretrained resnet-inception model is unfreezed while training the model for faces.

The trained model is used to predict the faces.

Note,Instead of training a neural network from scratch, it's better to start with a pre-trained network and then finetune it to recognize the face. 
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

#### 1.Amitabh Bacchan
#### 2.Ananya Pandey
#### 3.Anil Kapoor
#### 4.Bill Gates
#### 5.Elon Musk
#### 6.Tom Hanks
#### 7.Lion Messi
#### 8.Barrack Obama
#### 9.Micheal Jordan
#### 10.Lalu Yadav

There are 15 images of train data and 5 images of test data.


# Data Structure:
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
                   
               
