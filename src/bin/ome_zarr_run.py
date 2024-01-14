import fire
from src.process import Utilities as utils


# def _SeparateFlagArgs(args):
#     try:
#         index = args.index('--help')
#         args = args[:index]
#         index = args.index('-h')
#         args = args[:index]
#         return args, ['--help', '-h']
#     except ValueError:
#         return args, []
#
# fire.core.parser.SeparateFlagArgs = _SeparateFlagArgs

def apply_projection():
    _ = fire.Fire(utils.apply_projection_cmd)

