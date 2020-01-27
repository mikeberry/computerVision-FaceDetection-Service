from . import face_api_blueprint
from flask import json, jsonify, request
from FaceDetector import FaceDetector
import numpy as np
import cv2
import base64

face_detector = FaceDetector('dlib_face_recognition_resnet_model_v1.dat', 'shape_predictor_68_face_landmarks.dat')


@face_api_blueprint.route('/api/enroll', methods=['POST'])
def post_enroll():
    img = data_uri_to_cv2_img(request.json['img'])
    embeddings = face_detector.getEmbeddings(img)
    response = {}
    if(len(embeddings) >0):
        response = jsonify({"success":True})
    else:
        response = jsonify({"success":False,"message":"no face found"})
    return response

@face_api_blueprint.route('/api/queryFaces', methods=['POST'])
def query_faces():
    img = data_uri_to_cv2_img(request.json['img'])
    embeddings = face_detector.getEmbeddings(img)
    for i, embedding in enumerate(embeddings):
        embedding["embedding"]

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
