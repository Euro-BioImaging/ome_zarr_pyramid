import fire
from src.process import Utilities as utils

def apply_projection():
    _ = fire.Fire(utils.apply_projection)
    return 'finished'

