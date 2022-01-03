

Table Of Contents
-------------------------------------------
OVERVIEW
GOALS
TRAINING DATA PREPARATION
Creating dataset using MaskTheFace[2] and Flickr datasets[3]
Medical+mask[4] dataset
CNN Architecture
Training
Evaluation
References and Inspirations


OVERVIEW
--------------------------------------------
To create a Convolutional Neural Network (CNN) architecture using PyTorch and train the
model using datasets of images of following classes
1. Person without a face mask
2. Person with a “community” (cloth) face mask
3. Person with a “surgical” (procedural) mask, and
4. Person with an “FFP2”-type mask


GOALS
---------------------------------------------
1. To create a suitable Convolutional Neural Network (CNN) architecture using
PyTorch
2. Train the CNN architecture with datasets.


TRAINING DATA PREPARATION
----------------------------------------------
Dataset for this project is shared in the below location:(Not Public, only People with this
unique URL can access)
https://drive.google.com/drive/folders/1lsMegB_c6-NxLkBqj0xHYuJzXYIdUwbj?usp=sharing
People wearing cloth masks = 2035
People wearing ff2 mask = 995
People wearing no mask = 1001
People wearing surgical masks = 1091


The above dataset is prepared with the below resources

1. Creating dataset using MaskTheFace[2] and Flickr datasets[3]
Inspired by this data science article[1] we decided to prepare our dataset using
MaskTheFace[2] and Flickr datasets[3] .
Around 1000 images of human faces are downloaded from Flickr datasets[3] and using
MaskTheFace[2] scripts we have prepared a dataset for people wearing different types of
face masks. After execution of the scripts, we had 1000 images for each category.
2


Image : Screenshot of MaskTheFace script.Source: project team
Following is a typical output after we run MaskTheFace[2] on an image
Without mask Cloth Mask FFP2 Surgical
Image : output of MaskTheFace script. Source: project team
2. Medical+mask[4] dataset
This website provides Free to download and use datasets with JSON metadata for each
image in the datasets. This dataset has 6000 images with 20 classes of faces with
masks,no mask, medical mask, and partial mask images. We are using the below small
python script to read the metadata and arrange the images in specific folders. Also to
add flavours to the training dataset, we further group them to our training classes
3
Image : Script used for segregation of Medical+mask[4] images. Courtesy: Project Team.
Selected Images from this dataset are merged with the dataset created by
MaskTheFace[2]
Medical+mask[4] classes Mapped to this projects classes
Mask_surgical people_wearing_surgical_mask
Face_with_mask_incorrect
balaclava_ski_mask
people_wearing_no_mask
Mask_colorful
face_other_covering
people_wearing_cloth_mask
face_with_mask(further manually cleaned)
4
CNN Architecture
CNN is a special form of ANN which is used to recognize visual data as images .
It consists of 5 major parts :-
1. Convolution layer : A filter/kernel of 3*3 size runs across the image (32*32) taking
strides(amount of filter shift in each movement) and the resultant dot product is
saved , this process is called convolving . A padding of zeros (1) is added to the
image so as to prevent any data loss while this convolution The resultant matrix is
known as feature map.
We have one input Convolution layer and three hidden convolution layers .
Calculations :-
Input Image size = 32*32
Filter Size = 3*3
Padding = 1
Resultant Feature matrix => 32 - 3 + 2 * Padding + 1 =32 * 32
2. Pooling Layer :Since the amount of feature data is huge , the pooling layer pools
up the data from Convolution layers using any Max , Average , etc functions . we
have implemented two Max pool Layers .
Stride = 2
Kernel Size = 2
Resultant matrix of Max Pool Layer 1 => 16 * 16
Resultant matrix of Max Pool Layer 2 => 8 * 8
5
3. Fully Connected Layers : Here each node is connected to every other node , it
forms the last few layers of CNN . The input is flattened to one dimension and sent
to the last layer .
4. Activation Function : We used the Leakyrelu activation function. It defines when a
node will be fired; Leakyrelu is based on ReLU, Instead of the function being zero
when x < 0, a leaky ReLU will instead have a small negative slope (of 0.01, or so).
That is, the function computes f(x)=𝟙(x<0)(αx)+𝟙(x>=0)(x) where α is a small
constant
5. Optimization Function : We consider Adam for the optimization algorithm for
stochastic gradient descent for training deep learning models. Adam just adds the
expected value of the past gradient, so it can take different size steps for different
parameters and add with momentum for every parameter it can perform optimum.
Image : Screenshot of CNN model developed
6
Training
● Create a folder named PROJECT

● Download thes folderfrom

https://drive.google.com/drive/u/0/folders/1aodoJXPFfWkXCP3etN3F9gLz1UdyaMSt

● Copy the below folders to a folder named PROJECT/mask_dataset

● Copy testing_dataset_DO_NOT_COPY_TO_TRAINING_DATASET to a folder named
PROJECT/testing_dataset. Alternately, we can place the images which we want to
test in the folder testing_dataset in the corresponding known category

● For simplicity of path resolution, we will place the python file in the root folder
PROJECT

● Using our choice of IDE, or jupyter notebook, execute in the following order:

● Run command

AIFaceDetection('mask_dataset',None)

● This will create a model file named 'mask_dataset.pkl' in root folder PROJECT
Once the trained model is saved, we can use this for evolution using following command
AIFaceDetection('mask_dataset',None)


Summary of Facts of training
-------------------------------------------------------
7
Image Resized to 32 *32
No of Epoch 4
No of images in Training Split
No of images in Test Split 1535
Accuracy ~ .95
Evaluation
Image: Snapshot of the evaluated model
After training , we get the confusion matrix as above with the training/test split of
3584/1535 images . While testing , a new dataset is provided to the trained model , which
is able to give an accuracy of above 0.9 . We will further work in Phase Two of this CNN
implementation to increase the accuracy by including various optimizations on the raw
8
data as well as fiddling with various parameters of our CNN data . A more exhaustive
dataset will further give us a prediction closer to the real world application.
References and Inspirations
[1] https://towardsdatascience.com/creating-a-powerful-covid-19-mask-detection-tool-with-pytorc
h-d961b31dcd45
[2] https://github.com/aqeelanwar/MaskTheFace
License: MIT License
[3] https://github.com/NVlabs/ffhq-dataset
License: Creative Commons BY 2.0, Creative Commons BY-NC 2.0
[4] https://humansintheloop.org/resources/datasets/mask-dataset-download/?submissionGuid=85
8a270e-1056-4b68-9ed3-ddf3380516ce
License : Free to download and use, prepared by our humans in the loop
[5] Inspiration :
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-kerastensorflow-
and-deep-learning/
[6] Inspiration :
https://courses.analyticsvidhya.com/courses/take/convolutional-neural-networks-cnn-from-scra
tch/texts/10844923-what-is-a-neural-network
[7] Inspiration :
https://pytorch.org/docs/stable/generated/torch.nn.Module.html
[8] Testing Data
https://makeml.app/datasets/mask
License: CC0: Public Domain

     
