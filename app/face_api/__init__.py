from flask import Blueprint

face_api_blueprint = Blueprint('face_api', __name__)
from . import routes
