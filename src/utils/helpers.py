import logging, os, random, numpy as np
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    logger.info("Directory ensured at: %s", path)
