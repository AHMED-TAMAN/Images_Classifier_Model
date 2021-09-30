#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub



import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
import argparse
import time
from PIL import Image
import json


class_names = {}
def process_image(image, image_size=224):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,(image_size, image_size))
    image = tf.cast(image,tf.float32)
    image /= 255
    return image.numpy()

def predict(image_path, model, classes_labels, top_k=1, image_size=224):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image, image_size)
    expanded_image = np.expand_dims(processed_image, axis=0)
    probes = model.predict(expanded_image)
    top_k_values, top_k_indices = tf.nn.top_k(probes, k=top_k) # ref: https://stackoverflow.com/questions/50640687/get-top-k-predictions-from-tensorflow
   
    top_k_values = top_k_values.numpy().tolist()[0]
    top_k_indices = top_k_indices.numpy().tolist()[0]

    classes = [classes_labels[str(i+1)] for i in top_k_indices]
    return top_k_values, classes


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='Recognize different species of flowers images')
    parser.add_argument('img_path', help="path to the img")
    parser.add_argument('model', help="path to model HDF5 file")
    parser.add_argument('--top_k', help="num of top prediction probabilities", default=3)
    parser.add_argument('--category_names', help="path to images labels", default='label_map.json')
   
   
    args = parser.parse_args()
   
    image_path = args.img_path
    model=tf.keras.models.load_model('./{}'.format(args.model),custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(image_path, model, class_names, int(args.top_k))
   
    result = {}
    for i, (c, p) in enumerate(zip(classes, probs)):
        key = 'prob_{}'.format(i+1)
        result.update({key: {'class': c, 'probabilit': p}})
    print(result)





