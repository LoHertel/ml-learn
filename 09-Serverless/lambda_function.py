#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
from urllib import request

from numpy import divide
from PIL import Image

import tflite_runtime.interpreter as tflite


class DinoDragonPredictor:

    classes = ['dino', 'dragon']

    def __init__(self, model_path):
        
        # initialize model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

        # shape and image resolution, the model was trained for
        self.shape = self.interpreter.get_input_details()[0]['shape']
        self.image_resolution = self.shape[1:3]


    def preprocess_from_url(self, url):

        # load image from url
        with request.urlopen(url) as resp:
            buffer = resp.read()
        stream = BytesIO(buffer)
        img = Image.open(stream)
        
        # convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # resize image
        img = img.resize(self.image_resolution, Image.Resampling.NEAREST)
        
        # rescale and reshape data
        X = divide(img, 255, dtype='float32').reshape(self.shape)

        return X
    
    
    def predict(self, url):
        X = self.preprocess_from_url(url)

        self.interpreter.set_tensor(self.input_index, X)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_index)

        float_prediction = float(preds[0][0])

        result = {
            'probability_for_dragon': float_prediction, 
            'prediction': self.classes[round(float_prediction)] # prediction threshold 0.5 => Dino: [0, 0.5), Dragon: [0.5, 1]
        }

        return result



def lambda_handler(event, context):
    url = event['url']

    dino_dragon = DinoDragonPredictor(model_path='dino-vs-dragon-v2.tflite')

    result = dino_dragon.predict(url)
    return result
