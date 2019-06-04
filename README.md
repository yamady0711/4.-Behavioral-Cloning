# **Writeup for Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing that no tire leaves the drivable portion of the track surface

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
( In drive.py, I changed the driving set speed from "9 mph" to "20 mph". )
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.  
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 80-93)   
The model includes RELU layers to introduce nonlinearity (code line 93),  
and the data is normalized in the model using a Keras lambda layer (code line 81). 

#### 2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting (model.py lines 88).   
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

#### 4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road.  
I tried several methods of data collection and used all images collected(center, left, and right).  
I attempted to drive both clockwise direction and counter-clock-wise direction.
Also, I used a combination of center lane driving, recovering from the left and right sides of the road. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to essentially use models that were already created.
I took Nvidia model as most powerful model as suggested in the lesson.  
To avoid the overfitting, I modified Nvidia model by adding Dropout function of 50% rate.
Also, to avoid both underfitting and overfitting, I attempted sufficient amount of several type of collecting training data.

The final step was to run the simulator to see how well the car was driving around track one.  
At the end, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-93) consisted of a convolution neural network with the following layers and layer sizes ...
| Layer (type)                     | Output Shape        | Param #  | Connected to          |
| -------------------------------- |:-------------------:| --------:| --------------------: |
| lambda_1 (Lambda)                | (None, 160, 320, 3) | 0        | lambda_input_1  |
| cropping2d_1 (Cropping2D)        | (None, 65, 320, 3)  | 0        | lambda_1        |
| convolution2d_1 (Convolution2D)  | (None, 31, 158, 24) | 1824     | cropping2d_1    |
| convolution2d_2 (Convolution2D)  | (None, 14, 77, 36)  | 21636    | convolution2d_1 |
| convolution2d_3 (Convolution2D)  | (None, 5, 37, 48)   | 43248    | convolution2d_2 |
| convolution2d_4 (Convolution2D)  | (None, 3, 35, 64)   | 27712    | convolution2d_3 |
| convolution2d_5 (Convolution2D)  | (None, 1, 33, 64)   | 36928    | convolution2d_4 |
| dropout_1 (Dropout)              | (None, 1, 33, 64)   | 0        | convolution2d_5 |
| flatten_1 (Flatten)              | (None, 2112)        | 0        | dropout_1       |
| dense_1 (Dense)                  | (None, 100)         | 211300   | flatten_1       |
| dense_2 (Dense)                  | (None, 50)          | 5050     | dense_1         |
| dense_3 (Dense)                  | (None, 10)          | 510      | dense_2         |
| dense_4 (Dense)                  | (None, 1)           | 11       | dense_3         |

Total params: 348,219  
Trainable params: 348,219  
Non-trainable params: 0  


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![driving center of the road][./center.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center  
so that the vehicle would learn to  move back to center.  

Then I repeated this process on opposite direction to get more data points.  

To augment the data sat, I used left and right camera in addtion to center camera,  
moreover, I also flipped images and angles to further augmentation. 

After the collection process, I had 15,235 data points.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  

### Simulation
Please refer to the a video file(video.mp4) recording of my vehicle driving autonomously around the track. 


