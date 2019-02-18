# Darknet 

The darknet class is responsible of performing object segmentation on
the images/video passed to it.

The class is a child of the *Thread* so that its execution is
independent from the other classes and can handle execution without
terminating the entire program.

## Usage 


The class initialization is straightforward, it uses a boolean variable
for the choosing of the model to be used, either the one trained on the
[coco dataset](http://cocodataset.org/) or the one trained on the
[openimage dataset](https://storage.googleapis.com/openimages/web/index.html). 


**NB**: before using the Darknet class be sure to download the model'
weights using the [setup](setup.sh) script.

More over the threshold for the *minimum detection confidence* can be
set inside the class (*min_score*)
