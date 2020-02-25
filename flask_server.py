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

# initialize our Flask application and the Keras model
UPLOAD_FOLDER = 'static'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#cascPath = "haarcascade_frontalface_default.xml"
cascPath = os.path.join(app.config['UPLOAD_FOLDER'], 'haarcascade_frontalface_default.xml')

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'placeholder.png')
    
    return render_template('index.html', displayedimage = full_filename)
    
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        EMOTIONS = ["angry" ,"Fear", "happy", "sad", "surprised"]

        content = request.get_data(as_text = True)
        content = str(content)
                            
        image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], content))

        
        image = cv2.resize(image,(48,48))
        image = np.reshape(image,[1,48,48,3])
        
        prediction = model.predict(image)

        listprediction = prediction[0].tolist()
        maxidx = listprediction.index(max(listprediction))
                
        result = str(EMOTIONS[maxidx])
        
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], content))
        
        # below doesn't work properly - deleted before image loads for user
        # face_location = os.path.join(app.config['UPLOAD_FOLDER'], 'F' + content)
        # os.remove(face_location)
                        
    return result
    
@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            feedback = 'No file found.'
            return jsonify(msg = feedback, success = False)
        file = request.files['file']
        if file.filename == '':
            feedback = 'No file name.'
            return jsonify(msg = feedback, success = False)
        if file and allowed_file(file.filename):
        
            millis = int(round(time.time() * 1000))
            millis = str(millis)
        
            filename = millis + secure_filename(file.filename)
            f = request.files['file']
            image_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(image_location)
                        
            face_filename = 'F' + filename
                       
            imageFace = face_recognition.load_image_file(image_location)
            faces = face_recognition.face_locations(imageFace)

            print("Found {0} faces!".format(len(faces)))
            for (x, y, w, h) in faces:
                cv2.rectangle(imageFace, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if len(faces) == 1:
                feedback = '1 face found!'
            elif len(faces) == 0:
                feedback = 'No faces found - running analysis anyway.'
            else:
                feedback = 'More than 1 face found - running analysis anyway.'
            
            for face in faces:

                #Print the location of each face in this image
                top, right, bottom, left = face
                #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

                # You can access the actual face itself like this:
                face_image = imageFace[top:bottom, left:right]
                im = Image.fromarray(face_image)
            
                face_location = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                im.save(face_location)
            
            return jsonify(original_image = filename, face_image = face_location, msg = feedback, success = True)
                
        else:
            feedback = 'It seems you haven\'t uploaded an image. Your file must be of type jpg or png.'
            return jsonify(msg = feedback, success = False)
                                        
if __name__ == "__main__":

    print("Loading Keras model and Flask starting server...")
                
    model = load_model('Trained_Model.h5')
    
    print('model loaded')
    
    app.run(host = "127.0.0.1", port = 5001, debug = False, threaded = False)

# ps -fA | grep python
# kill -9 pid