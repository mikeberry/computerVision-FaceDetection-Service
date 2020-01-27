from flask import Flask
from face_api import face_api_blueprint


def create_app():
    app = Flask(__name__)
    app.register_blueprint(face_api_blueprint)

    return app
