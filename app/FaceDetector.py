import dlib
import cv2
import numpy as np


class FaceDetector:
    def __init__(self, path_face_recognition_model, path_face_landmark_model):
        self.faceDetector = dlib.get_frontal_face_detector()
        self.shapePredictor = dlib.shape_predictor(path_face_landmark_model)
        self.faceRecognizer = dlib.face_recognition_model_v1(path_face_recognition_model)

    def getEmbeddings(self, img):
        face_embeddings = []
        faces = self.faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for k, face in enumerate(faces):
            shape = self.shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
            face_pos = [face.left(), face.top(), face.right(), face.bottom()]

            faceDescriptor = self.faceRecognizer.compute_face_descriptor(img, shape)
            faceDescriptorList = [x for x in faceDescriptor]
            faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
            faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

            face_embeddings.append({"face_pos": face_pos, "embedding": faceDescriptorNdarray})

        return face_embeddings
