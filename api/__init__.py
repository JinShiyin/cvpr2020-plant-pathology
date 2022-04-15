from flask import Blueprint
from flask_cors import CORS
from flask_restplus import Api

from .photos import api as ns_photos
blueprint = Blueprint('api', __name__, url_prefix='/api')

cors = CORS(blueprint)

api = Api(
    blueprint,
    title='OpticalCanding',
    version='v1',
)

# mount related blueprint
api.add_namespace(ns_photos)