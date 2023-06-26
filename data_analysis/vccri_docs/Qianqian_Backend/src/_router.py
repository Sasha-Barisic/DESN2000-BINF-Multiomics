from flask import Blueprint, request, send_file
from flask_json import JsonError, json_response, as_json
from werkzeug.utils import secure_filename
from src._global import *
import os

router_api = Blueprint('router_api', __name__)

@as_json
@router_api.route("/", methods=['GET'])
def index():
    return json_response(data=[1,2,3], hello='xixi')


@router_api.route("/get_json",  methods=['POST'])
def get_json():
    data = request.get_json()
    print(data['pwd'], data['name'])
    return json_response(data=[1,2,3], hello='xixi')

@router_api.route("/file", methods=['POST'])
def upload_file():
    f = request.files['file']
    print(request.files['file'])
    f.save(UPLOAD_FILE_PATH + secure_filename(f.filename))

    return "1"

@as_json
@router_api.route("/get_file/<file>", methods=['GET'])
def get_file(file):
    path = CLEAN_DATA_PATH + secure_filename(file)
    if os.path.exists(path):
      return send_file(path);
    else:
      raise JsonError(err='file not exits')