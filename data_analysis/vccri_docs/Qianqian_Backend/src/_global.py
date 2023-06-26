import os

CURRENT_PATH = os.getcwd()
ASSETS_PATH = os.path.join(CURRENT_PATH, 'assets')
UPLOAD_FILE_PATH = os.path.join(CURRENT_PATH, 'assets/upload/')
CLEAN_DATA_PATH = os.path.join(CURRENT_PATH, 'assets/clean/')

if not os.path.exists(ASSETS_PATH):
    os.mkdir(ASSETS_PATH)

if not os.path.exists(UPLOAD_FILE_PATH):
    os.mkdir(UPLOAD_FILE_PATH)

if not os.path.exists(CLEAN_DATA_PATH):
    os.mkdir(CLEAN_DATA_PATH)