## Lane Detection with Deep Learning
![A](Graph/result_34500_tensor(1.8018).png)
### Project Overview

When a human drive a car, the most common task is to keep the car in the traffic lane. As long as the driver has not been distracted while driving, this task is easy and possible for someone with basic training. On the other hand, for computers this task, keeping the car between its lane’s lines, is not as easy as human.

The reason behind this difficulty is that, computers do not have an ability to understand the environment they are in. The yellow and white marking lanes are not understandable for computers inherently. Thank to various kinds of techniques such as computer vision and deep learning, this task would be teachable and possible for computers to understand.

### Why this matter?

Fully autonomous driving relies on the understanding the environment around the vehicle. Various perception modules are used for this understanding, and many pattern recognition and computer vision techniques are applied for these perception modules. Lane detection, which identifies the drivable area on a road, is a major perception technique required for autonomous driving.

### Problem Statement

It is true that human beings can identify lane line markings while driving with basic training, but sometimes, based on the number of crashes and accidents, it is understandable that they can have a disadvantage of not always being attentive. Although, it is not that easy for computers to learn identifying lane line markings but after learning the task, there would not be any distractions for them and they have this advantages that the rate for crashes and accidents caused by distraction would be less and computers can take over this task from human driver. Using deep learning, I have used transfer learning and re-train a pre-trained model (PINet ). The model is based on a Convolutional Neural Network (CNN) which perform well on image dataset. CNNs work well with images by looking at them in pixel level. 

### Dataset 

Since the dataset (images from Maryland Department of Transportation) is unlabeled, I had to use labeled public dataset (Tusimple ) to retrain the model. This contains 3626 video clips of 1 sec duration each. Each of these video clips contains 20 frames of which, the last frame is annotated. These videos were captured by mounting the cameras on a vehicle dashboard.

You can download the dataset from https://github.com/TuSimple/tusimple-benchmark/issues/3. It has been advised to use the structure directory as below:

    dataset
      |
      |----train_set/               # training root 
      |------|
      |------|----clips/            # video clips, 3626 clips
      |------|------|
      |------|------|----some_clip/
      |------|------|----...
      |
      |------|----label_data_0313.json      # Label data for lanes
      |------|----label_data_0531.json      # Label data for lanes
      |------|----label_data_0601.json      # Label data for lanes
      |
      |----test_set/               # testing root 
      |------|
      |------|----clips/
      |------|------|
      |------|------|----some_clip/
      |------|------|----...
      |
      |------|----test_label.json           # Test Submission Template
      |------|----test_tasks_0627.json      # Test Submission Template

### Fundemantal of Current Model
Reference Paper: https://arxiv.org/abs/2002.06604 
Deep learning methods show outstanding performance for complex scenes. Among deep learning methods, Convolutional Neural Network (CNN) methods are primarily applied for feature extraction in computer vision.

Lane-mark detection and road region segmentation are the two most popular and common techniques for identifying lanes. The main goal of segmentation is to partition an image into regions. This project uses lane-mark detection for lane identification. 

Semantic segmentation methods, the major research area in computer vision, are frequently applied to traffic line detection problems to make inferences about shapes and locations. Some methods use multiclass approaches to distinguish individual traffic line instances. Therefore, even though these methods can achieve outstanding performance, they can only be applied to scenes that consist of fixed numbers of traffic lines. As a solution to this problem, instance segmentation methods are applied to distinguish individual instance by making clusters.

The method that has been used for this project is based on the key points estimation and instance segmentation approach. Key-points are the same thing as interest points. They are spatial locations, or points in the image that define what is interesting or what stand out in the image.


### Result and Conclusion
In this project, the PINet lane detection model has been re-trained with TuSimple dataset. This lane detection method combined key point estimation and point instance segmentation methods. This model achieves high performance and a low rate of false positives; And as we know, false positives could cause major accidents and the lowest the rate the more the safety performance of the autonomous vehicle would be. The table below shows results for this project:

| Accuracy | FP   | FN   |
| -------- | ---- | ---- |
| 96.75%   |0.030 |0.020 |

![Accuracy Trend](Graph/Picture1.png)

The trend below also shows how the training improved within each epoch.

![Epoch Improvement Visulization](Graph/improvement.png)


The images below are the final result of testing the model on the unlabeled dataset (Images of Maryland roads) and also the attached link is the video from route MD3 in Anne Arundel County. 

### Future Work
Expanding the model to detect even more items at the same time would be very useful for autonomous transportation system. Detecting the lane line markings, vehicles and pedestrian at the same time is a great task for future of , lane, object detection and image segmentation. 

### References:
1.	https://medium.com/analytics-vidhya/detecting-lanes-using-deep-neural-networks-eebf2d9e3603

2.	https://arxiv.org/abs/2002.06604
