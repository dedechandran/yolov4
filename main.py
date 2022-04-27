from flask import Flask, request, jsonify, Response
import app.preprocess as prep
import app.detector as detector

app = Flask(__name__)

@app.route("/")
def home_view():
    return "<h1>goblog</h1>"

@app.route("/detect",methods =['POST'])
def detect_object():
    request_body = request.get_json()
    encoded_bitmap = request_body['encoded_image']
    decoded_image = prep.decodeBase64Image(encoded_bitmap)
    detected_objects = detector.detect_object(decoded_image) 
    return jsonify(
        images = detected_objects
    )