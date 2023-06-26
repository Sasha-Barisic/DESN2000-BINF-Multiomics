from flask import Blueprint, request, send_file
from flask_json import JsonError, json_response, as_json
from werkzeug.utils import secure_filename
from src._global import *
import os

file_router_api = Blueprint('file_router_api', __name__)

@file_router_api.route("/upload_file", methods=['POST'])
def upload_file():
    f = request.files['file']
    # print(request.files['file'])
    f.save(UPLOAD_FILE_PATH + secure_filename(f.filename))
    return json_response(ok=True)

@as_json
@file_router_api.route("/get_file/<file>", methods=['GET'])
def get_file(file):
    if os.path.exists(UPLOAD_FILE_PATH + secure_filename(file)):
      return send_file(UPLOAD_FILE_PATH + secure_filename(file))
    else:
      raise JsonError(err='file not exits')