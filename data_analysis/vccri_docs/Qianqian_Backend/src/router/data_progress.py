from flask import Blueprint, request, send_file
from flask_json import JsonError, json_response, as_json
from werkzeug.utils import secure_filename
from src._global import *
import os
from src.data.clean import data_clean
import pandas as pd

data_progress_router_api = Blueprint('data_progress_router_api', __name__)

@as_json
@data_progress_router_api.route("/data_clean/<file>", methods=['GET'])
def clean_data(file):
    path = UPLOAD_FILE_PATH + secure_filename(file)
    print(path)
    if os.path.exists(path):
      df = pd.read_excel(path)
      ndf = data_clean(df)
      print(ndf.shape)
      return json_response(columns=ndf.columns, DataFrame=ndf.to_numpy())
    else:
      raise JsonError(err='file not exits')
