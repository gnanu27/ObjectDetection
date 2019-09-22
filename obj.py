from flask import Flask, render_template, request
from keras.applications.inception_v3 import *
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import json
import io

app = Flask(__name__)

model = InceptionV3(weights='imagenet')
# graph = tf.get_default_graph()
# def predict(path):
#     img = image.load_img(path, target_size=(299,299))
#     xy = image.img_to_array(img)
#     xy = np.expand_dims(xy, axis=0)
#     xy = preprocess_input(xy)
#     global graph
#     with graph.as_default():
#         preds = model.predict(xy)
#     preds = decode_predictions(preds, top=3)[0]
#     acc = []
#     classes = []
#     for x in preds:
#         acc.append(x[2])
#         classes.append(x[1])
#     return acc, classes

# print(predict('./resto-back.jpg'))
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('obj.html')
    
    if request.method == 'POST':
        if request.files.get('fileImage'):
            path = request.files["fileImage"]
            # path = Image.open(io.BytesIO(path))
            print('Hello',path)
            img = image.load_img(path, target_size=(299,299))      
            xy = image.img_to_array(img)
            xy = np.expand_dims(xy, axis=0)
            xy = preprocess_input(xy)
            global graph
            with graph.as_default():
                preds = model.predict(xy)
            preds = model.predict(xy)
            preds = decode_predictions(preds, top=3)[0]
            acc = []
            classes = []
            for x in preds:
                acc.append(x[2])
                classes.append(x[1])
            return render_template('obj.html', preds = acc, classes = json.dumps(classes), img=path)
        else:
            return render_template('obj.html', preds = .5)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)