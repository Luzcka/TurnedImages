Train a ML to recognize if a person picture is in the right position or if it's turned left, turned right or upside down.


IMPORTANT

 1) Fixed image files are inside "myoutput.tar.gz" file
 2) CSV file with the test execution is "test.preds.csv"
 3) The Keras model was saved in turnedimages.h5 but it was compressed because the size. Uncompress before run.

 4) Create "output" directory
 5) Train and Test images was placed in "train" and "test" folders respectivelly.



This solution was implemented with Python 3.6 inside a Conda environment using these dependences:

Tensorflow
Keras
NumPy
Pillow
Pandas
scipy
scikit-learn


Was implemented two scripts called: trainme.py and testme.py. The firt one has the trainning routines to generate a model that is save in the same directory of the script, the model file name is turnedimages.h5, and the second script will make use of this model to classify new files predicting if them are turned or not.

To Train:

Parameter 1: Directory where area located the images to train
Parameter 2: The CSV file with the info to train composed by : file name and Label

execute in a terminal: python trainme.py ./train ./train.truth.csv


To Test / Predict:

Parameter 1: Directory where area located the images to train
Parameter 2: The CSV file with the info to train composed by : file name and Label

execute in a terminal: python testme.py ./test ./test.preds.csv ./output

Obs.: You need a folder called "output" like in the GitHub project.


Solution and Theory:

First of all I chose Keras to solve this problem, not because a special technology or scientific aproach, but because I'm a user of this solution. At firt time I had thought to use SVM because it's a common classification but I never used SVM with images. I chose the secure aproach because the limited time.

To read the trainig data was used "different flow_from_dataframe" method of ImageDataGenerator, this mthod let us read the data information (file name and Label) from a file like a CSV, creating a Pandas DataFrame that is going to be used by Keras, another aproach is the use of "flow_from_directory", but in this case the images need to be "classified" by subdirectories, it was not my case. I opted by preserve the image size and others attributes because by a superficial analysis the images were in good shape. The model choosed was Sequential, it's easy to define, use and make some tunning, allowing us to add layer by layer. Each layer has weights that correspond to the layer the follows it. in the case os a Dense type layer was defined with 512 nodes because the use of images, the class mode was defined as "categorical" because it is a classification task.

At the last the model was compiled and evaluated, to make an evaluation I define that 25% of the images for trainig was destinated to evaluation, because this this trainig didn't use the 100% of train images. At the end "epochs" was changed from 10 to 2 because the time, I had problems and lost a several time, I can say that with "epochs" equal 10 in my first text the model achived more than 95%, with the value equal 2 this was reduced to 92%. After the model validation it is saved in the root project directory to be used by the test script.

The test script only read the model, compile it and apply to the test directory creating a CSV file with the prediction and saving the fixed files to an output directory.


Training result:

Using TensorFlow backend.
Found 36672 images belonging to 4 classes.
Found 12224 images belonging to 4 classes.
Epoch 1/2
7334/7334 [==============================] - 1708s 233ms/step - loss: 0.4293 - acc: 0.8334 - val_loss: 0.2336 - val_acc: 0.9179
Epoch 2/2
7334/7334 [==============================] - 1523s 208ms/step - loss: 0.2377 - acc: 0.9192 - val_loss: 0.1738 - val_acc: 0.9394



Some install commands:

conda create -n turnedimages python=3.6
source activate turnedimages

conda install -n turnedimages scipy numpy pandas scikit-learn pillow

pip install --upgrade tensorflow
pip install Keras