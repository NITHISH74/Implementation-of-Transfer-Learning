# EX04--Implementation-of-Transfer-Learning
## Aim:
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset:
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:
![237872257-0c867e5a-02f9-4e55-b90e-0f50b649c456](https://github.com/sithihajara/Implementation-of-Transfer-Learning/assets/94219582/3b5490e5-300b-4cf7-8556-6e4198e10acd)

## DESIGN STEPS
### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Load CIFAR-10 Dataset & use Image Data Generator to increse the size of dataset

### STEP 3:
Import the VGG-19 as base model & add Dense layers to it
### STEP 4:
Compile and fit the model

### STEP 5:
Predict for custom inputs using this model


## PROGRAM:
```python
### Developed by    : Nithishwar S
### Register number : 212221230071
```
```python
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG19 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data ()
x_train = x_train.astype ("float32") / 255.0
x_test = x_test.astype ("float32") / 255.0
base_model = VGG19 (include_top = False, input_shape = (32, 32, 3))

for layer in base_model.layers:
  layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense (512, activation = 'relu'))
model.add(Dropout (0.5))
model.add(Dense (10, activation= 'softmax'))

model.compile (optimizer = Adam (learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics='accuracy'
              )

learning_rate_reduction = ReduceLROnPlateau (monitor = 'val_accuracy')
metrics = pd.DataFrame (model.history.history)
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(x_test), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```


## OUTPUT:
### Training Loss, Validation Loss Vs Iteration Plot
Training Loss, Validation Loss Vs Iteration             | Accuracy, Validation Accuracy Vs Iteration                   |               
:------------------------------------------------------:| :-----------------------------------------------------------:|
| ![1](https://github.com/Venkatigi/Implementation-of-Transfer-Learning/assets/94154252/f13afc27-551a-4048-ac2a-03860b85de5e)| ![2](https://github.com/Venkatigi/Implementation-of-Transfer-Learning/assets/94154252/16a25f9d-4a36-4275-9532-0d391937c803)|
### Classification Report:
![4](https://github.com/Venkatigi/Implementation-of-Transfer-Learning/assets/94154252/8d5036ac-3064-4df2-8298-88e9899954cd)

### Confusion Matrix:
![3](https://github.com/Venkatigi/Implementation-of-Transfer-Learning/assets/94154252/9a1ddaef-45dd-4332-91cf-61dacf781516)

## Conculsion:
* We got an Accuracy of 60% with this model.There could be several reasons for not achieving higher accuracy. Here are a few possible explanations:
### Dataset compatibility: 
* VGG19 was originally designed and trained on the ImageNet dataset, which consists of high-resolution images. 
* In contrast, the CIFAR10 dataset contains low-resolution images (32x32 pixels). 
* The difference in image sizes and content can affect the transferability of the learned features. 
* Pretrained models like VGG19 might not be the most suitable choice for CIFAR10 due to this disparity in data characteristics.

### Inadequate training data: 
* If the CIFAR10 dataset is relatively small, it may not provide enough diverse examples for the model to learn robust representations. 
* Deep learning models, such as VGG19, typically require large amounts of data to generalize well. 
* In such cases, you could consider exploring other architectures that are specifically designed for smaller datasets, or you might want to look into techniques like data augmentation or transfer learning from models pretrained on similar datasets.

### Model capacity: 
* VGG19 is a deep and computationally expensive model with a large number of parameters. 
* If you are limited by computational resources or working with a smaller dataset, the model's capacity might be excessive for the task at hand. 
* In such cases, using a smaller model architecture or exploring other lightweight architectures like MobileNet or SqueezeNet could be more suitable and provide better accuracy.



## RESULT:
Thus, transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture is successfully implemented.
