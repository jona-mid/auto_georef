"""
Configuration for georeferencing quality check system.
"""

# Viewport settings
VIEWPORT_SIZE = 1024
ZOOM_LEVEL = 17
BASEMAP_STYLES = ["satellite"]  # Start with satellite, can add more

# deadtrees.earth API
DEADTREES_API_URL = "https://deadtrees.earth/api/v1"
DATASET_ID = 1  # Set to your dataset ID

# Synthetic negative generation params
OFFSET_RANGE_M = [-200, 200]  # meters
ROTATION_RANGE_DEG = [-10, 10]
SCALE_RANGE = [0.95, 1.05]
HARD_NEGATIVE_OFFSET_M = 500  # offset for hard negatives

# Training params
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Classifier params
CLASSIFIER = "xgboost"
THRESHOLD = 0.5

# Paths
DATA_DIR = "data"
RAW_DIR = DATA_DIR + "/raw"
PROCESSED_DIR = DATA_DIR + "/processed"
MODELS_DIR = DATA_DIR + "/models"

# Tile server (deadtrees.earth basemap)
TILE_SERVER_URL = "https://deadtree-ai.github.io/deadtrees/tiles"
