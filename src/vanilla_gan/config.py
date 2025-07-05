import yaml
import os
import uuid

UNIQUE_RUN_ID = str(uuid.uuid4())

# Get the config in project root dir
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)