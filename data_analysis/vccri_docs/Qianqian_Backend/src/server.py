from flask import Flask
from src.router.file import file_router_api
from src.router.data_progress import data_progress_router_api
from flask_json import FlaskJSON

app = Flask(__name__)
json = FlaskJSON(app)

app.config['JSON_ADD_STATUS'] = True
app.config['JSON_DATETIME_FORMAT'] = '%d/%m/%Y %H:%M:%S'

# app.register_blueprint(router_api)
app.register_blueprint(file_router_api)
app.register_blueprint(data_progress_router_api)

def server(port, host, debug = True):
    
    app.run(port=port, host=host, debug=debug)