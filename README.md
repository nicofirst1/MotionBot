# MotionBot

This branch brings the concept of smart surveillance system beyond its original scope. 
The famous [Darknet](https://pjreddie.com/darknet/) framework has been successfully integrated, making the object recognition 
task fast and secure.\
More over the face recognition task has been improved using the
 [face recognition repo](https://github.com/ageitgey/face_recognition).

## Installation
In this section the setup for this branch is explained.\
You **must** perform the setup of the master branch first in order to make this one work.


### Python 
Install the requirements using\
`pip install -r requirements.txt`\
So that the additional packages will be installed.

### Darknet Setup
For the darknet installation refer to the following links:
- Follow the instructions in the [darknet](https://pjreddie.com/darknet/install/) to compile the darknet environment
- Run [setup.sh](./setup.sh) in the main directory to perform the setup


## Usage

There are various python files which can be executed.

### Main 

To start the secure system first position your camera where you want and then use the [main.py](./src/main.py) file:\
`python ./src/main.py`\

If everything worked fine you should see the following messages:
```
TOKEN : your token
Password : 12345678
getting ids from file
...
Darknet metadata loading
...
camera connected
Polling...
Reset ground image ...
Done
```
If not look at the [Journal](./Journals/Journal_Darknet.md) to check for possible solutions to your problem.

### Tests
There are multiple kind of test executables in the [Test](./src/Tests) directory.

#### Darknet 

[This file](./src/Tests/darknet_test.py) works uniquely with the darknet framework showing how it works in real time. You can also take screenshots of what 
the program is seeing (notice that a webcam must be connected in order to make it work), as shown in the following image.
![Darknet example](./Resources/Readme/darknet_test.jpg)

#### Photo recognizer
[This file](./src/Tests/photo_recognizer_test.py) is used to test the recognizer with static images. It supports the loading of a custom 
image from the device and the choosing of multiple types of algorithm for face recognition 
(see the [Classification algorithms section](#classification_alg) for more information on the algorithms).
![Photo example](./Resources/Readme/photo_reco.png)


#### Video recognizer
[This file](./src/Tests/video_recognizer_test.py) does the same as the previous two but on videos. It allows to segment the 
video feed from the webcam with both the face recognizer and darknet. Notice that the activation of both will slow down the cam FPS.
![Video example](./Resources/Readme/video_reco.png)
Moreover there are some additional functionality:
- The *save faces* checkbox is used to save detected faces in the [Unkown](./Faces/Unknown) directory in order to gain 
samples faster than using the surveillance system.
- The *clean face* button is used to filter those images in the [Unkown](./Faces/Unknown) directory which have a similarity higher than a 
certain threshold
- The *Train* button trains the model on new images in the [Faces](./Faces) directory. 


## Face Recognizer

### <a name="classification_alg"></a> Classification algorithms
The classification algoritmhs are chosen using the *clf_flag* in the [Face recognizer](./src/Classes/Face_recognizer.py) file. 
Each integers maps to a different method:
- 0 -> SVM
- 1 -> KNN
- 2 -> Distance : top n
- 3 -> Distance : minimum sum 

### Dataset
The dataset is made of the face embeddings and an associated list of the people they belongs to. These embeddings, called *encodings* 
in the [face recognition repo](https://github.com/ageitgey/face_recognition) , are arrays of 128 floats. They can be seen as 
"summaries" extracted from a person's face ([here](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
are more information about how the embeddings extraction works).\
Using PCA on the a dataset made of two individuals with 124 and 144 embeddings respectively yelds the following plots:

2d plot           |  3d plot
:-------------------------:|:-------------------------:
![Plot 1](./Resources/Readme/2d_plot.png)  | ![Video example](./Resources/Readme/3d_plot.png)

Moreover the variance per componetns can be plotted too, which will be useful for the KNN algorithm
![Variance](./Resources/Readme/variance_components.png)
<img src="./Resources/Readme/variance_components.png" alt="Variance" width="200"/>


### SVM
This is the easiest classificator since it fits the dataset and then performs a prediction based on what it has learned.
No additional configuration is needed, but some tuning may increase its accuracy. Moreover the [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
does not provide a measure of probability for its decision, hence the confidence in the prediction is set to a constant "-".
 
### KNN
This is the approach used by *ageitgey* in its [example code](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py) where
a k-nearest-neighbors classifier is used. Parameters such as the number of neighbors can be tweaked directly in the *predict* function.

### Distance 
This category of custom algorithm make use of the [face distance](https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L60)
function which returns an array of distances given the train set and a new face encoding.\
A *normalized distance vector*, which is the distance vector normalized so that its elements sums up to 1, is used in 
the following computations.

#### Top n
The *top n* algorithm implement a voting system based on taking the top *n* closest distances to the one to predict. 
Then it counts the number of times each label appears in the vector of lenght *n* and return the most common one.

#### Lowest Sum
This algorithm uses the normalized distance vector to sum all the distances belonging to a specific label and then 
normalize them with the number of times that labels appears in the dataset. It then return the label with the lowest 
value.