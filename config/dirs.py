import os

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))


DATA_PATH = os.path.join(PROJECT_PATH,'Datasets')
MODELS_PATH = os.path.join(PROJECT_PATH,'Models')
RESULT_FILE_PATH = os.path.join(PROJECT_PATH,'Results')

CALIBRATED_MODELS_PATH = os.path.join(PROJECT_PATH,'Models')
SAVED_SEG_MAPS = os.path.join(PROJECT_PATH,'Segmentation_outputs')