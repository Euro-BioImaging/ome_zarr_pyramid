import fire
from ome_zarr_pyramid.process import array_manipulation as manip

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
    _ = fire.Fire(manip.apply_projection_cmd)

