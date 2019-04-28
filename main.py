from sys import platform as _platform
import numpy as np
from six.moves import urllib
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

PWD = '/var/www/html/flaskapp'
MODEL_PATH = PWD + '/data/output_graph.pb'
LABELS_PATH = PWD + '/data/output_labels.txt'


def create_graph():
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_url):
    # load image from url
    req = urllib.request.Request(image_url)
    response = urllib.request.urlopen(req)
    image_data = response.read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # get top 5 predictions
        top_k = predictions.argsort()[-5:][::-1]
        f = open(LABELS_PATH, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        results = []

        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]

            results.append([human_string, score])
            print('%s (score = %.5f)' % (human_string, score))

        return results


# HTTP API
app = Flask(__name__)

@app.route('/api/photo-recognize', methods=['POST'])
def photoRecognize():
    answer = run_inference_on_image(request.form['image_data'])
    // TODO: do some logic with answer here
    return jsonify(status='OK', results=answer)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
