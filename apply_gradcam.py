# import the necessary packages
from grad_dir.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from PIL import Image 
import PIL 
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=("vgg", "resnet"),
	help="model to be used")
args = vars(ap.parse_args())

# initialize the model to be VGG16
Model = VGG16

# check to see if we are using ResNet
if args["model"] == "resnet":
	Model = ResNet50

# load the pre-trained CNN from disk
print("[INFO] loading model...")
model = Model(weights="imagenet")

# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions
orig = cv2.imread(args["image"])
resized = cv2.resize(orig, (224, 224))

#saving name of image
name_img=os.path.split(args["image"])[-1]
print (name_img)

# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
image = load_img(args["image"], target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

# use the network to make predictions on the input image and find
# the class label index with the largest corresponding probability
preds = model.predict(image)
i = np.argmax(preds[0])

# decode the ImageNet predictions to obtain the human-readable label
decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))

# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)

# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
output_orig = output.copy()

#increasing the contrast of the heatmap
# heatmap = cam.contrast_controller(heatmap)
# print('type is ')
# print(type(heatmap))

# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.8, (255, 255, 255), 2)

#saving the image and heatmaps
#defining a path:
path = 'C:/Users/singhmay/workingdir/gradcam/Results/' + name_img[:-4]
#check if directory exists or not:
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
    print('Creating directory...')
#saving the image to the path
cv2.imwrite(os.path.join(path , Model.__name__ + '_original_' + name_img) , orig)
cv2.imwrite(os.path.join(path , Model.__name__ + '_output_' + name_img) , output_orig)
cv2.imwrite(os.path.join(path , Model.__name__ + '_heatmap_' + name_img) , heatmap)
cv2.imwrite(os.path.join(path , Model.__name__ + '_labelled_op_' + name_img) , output)
print('image saved to directory: ' , path)

# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)

#showing the image
cv2.imshow("Output", output)
cv2.waitKey(0)