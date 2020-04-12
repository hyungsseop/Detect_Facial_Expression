import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import Flask, request, redirect, url_for, render_template, jsonify, json
import io
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
import face_recognition
import cv2
import sys
import glob
import time
import sys
import pyrebase
import zmq
import time


UPLOAD_FOLDER = 'static'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

EMOTIONS = ["angry" ,"Fear", "happy", "sad", "surprised"]

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#cascPath = "haarcascade_frontalface_default.xml"
cascPath = os.path.join(app.config['UPLOAD_FOLDER'], 'haarcascade_frontalface_default.xml')

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

@app.route('/', methods=['GET', 'POST'])
def home():
    
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'placeholder.png')
    data = "please send image"
    return render_template('indextest.html', displayedimage = full_filename, data=data)

@app.route("/file", methods=["GET", "POST"])
def single_file():
    data = ""
    # request.content_length returns the content length in bytes
    content_length = request.content_length
    print(f"Content length: {content_length}")

    # content_type
    content_type = request.content_type
    print(f"Content type: {content_type}")

    # request.mimetype returns the mimetype of the request
    mimetype = request.mimetype
    print(mimetype)

    # Get an ImmutableMultiDict of the files
    file = request.files
    print(file)
    print(file.keys())
    # Get a specific file using the name attribute
    #if request.files.get("image"):
    if 1==1:
        
        #image = request.files["image"]
        #print(f"Filename: {image.filename}")
        #print(f"Name: {image.name}")
        #print(image)
        #image.save('/home/hyungseop/test/Images/unity.png')
        imageFace = cv2.imread('/home/hyungseop/test/static/12.png')

        faces = face_recognition.face_locations(imageFace)
        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(imageFace, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for face in faces:
            #Print the location of each face in this image
            top, right, bottom, left = face
            #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            # You can access the actual face itself like this:
            face_image = imageFace[top:bottom, left:right]
            #img = Image.fromarray(face_image)
            img = cv2.resize(face_image,(48,48))
            img = np.reshape(img,[1,48,48,3])
            ar=model.predict(img)
            ar = ar[0]
            for i in range(len(ar)):
                ar[i] = round(ar[i], 3)
            print(ar)
            iar = ar.tolist()
            maxidx = iar.index(max(iar))
            print(EMOTIONS[maxidx])
            data += EMOTIONS[maxidx] + " "
        # To save the image, call image.save() and provide a destination to save to
        # image.save("/path/to/uploads/directory/filename")
        print(data)
        data = str(data)

        #return render_template("indextest.html", data=data)

    return data


if __name__ == "__main__":

    print("Loading Keras model and Flask starting server...")
                
    model = load_model('Trained_Model.h5')
    
    print('model loaded')

    app.run(host = "0.0.0.0", port = 5001, debug = False, threaded = False)#, ssl_context='adhoc')

# ps -fA | grep python
# kill -9 pid
# ssl_context='adhoc' ,
