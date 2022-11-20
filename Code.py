from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
# Import Data Science Libraries
import numpy as np
from skimage.filters import threshold_otsu,gaussian

import pandas as pd
from skimage.util import crop
from skimage import color
from skimage.transform import resize
import cv2

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Tensorflow Libraries

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing

# System libraries
from pathlib import Path
import os.path



try:
    import shutil
    shutil.rmtree('uploaded / image')

    # print()
except:
    pass

model = tf.keras.models.load_model('model_best_2.h5')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/images'


def save_image(image, directory, filename):
    """
    Save an image to a new directory
    """
    cv2.imwrite(os.path.join(directory, filename), image*255)



@app.route('/',methods = ['GET'])
def upload_f():
    return render_template('index.html')
    # return "Hello"
def finds():
    # global labels
    # print("HEy")
    dataset = "uploaded/"
    image_dir = Path(dataset)  # Get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(
        image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.p.png'))

    # labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    # filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    # labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    # image_df = pd.concat([filepaths, labels], axis=1)
    test_generator = ImageDataGenerator( 
       rescale=1./ 255,
    )
    # test_df = image_df
    dir=r"C:/Users/shiji/Downloads/ji - Copy-20221119T203113Z-001/ji - Copy/uploaded/"
    print("Dir:",dir)
    test_images = test_generator.flow_from_directory(
        directory=dir,
       
        target_size=(512, 512),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    labels = {
        0:"Abnormal heartbeat",
        1:"History of MI",
        2:"MI",
        3:"NORMAL",
    }
    pred = model.predict(test_images)
    pred = np.argmax(pred, axis=1)

    # Map the label
    # labels = (labels)
    # labels = dict((v, k) for k, v in labels.items())
    # pred = [labels[k] for k in pred]
    print("Prediction:",pred)
    print("Prediction:",labels[pred[0]])
    # Display the result
    return(labels[pred[0]])


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':	
        f = request.files['file']
        # print("Hey-----", f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], "pic.jpg"))
        filepath=os.path.join(app.config['UPLOAD_FOLDER'], "pic.jpg")
        a = cv2.imread(filepath)

        y=crop(a, ((300, 100), (50, 50), (0,0)), copy=False)    
        grayscale = color.rgb2gray(y)
        grayscale=resize(grayscale,(1572,2213))
                #smoothing image
        blurred_image = gaussian(grayscale, sigma=1)
                #thresholding to distinguish foreground and background
                #using otsu thresholding for getting threshold value
        global_thresh = threshold_otsu(blurred_image)

                #creating binary image based on threshold
        binary_global = blurred_image < global_thresh
       
        print( os.path.join(app.config['UPLOAD_FOLDER'], "pic.jpg"))
        save_image(binary_global, os.path.join(app.config['UPLOAD_FOLDER']), "pic.jpg")

        print("raees")
        val = finds()
        print(val,"-----")
        return str(val)

if __name__ == '__main__':
    app.run(port=5001,debug=True)
