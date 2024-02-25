import fire
from ome_zarr_pyramid.process import image_operations as imop

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
    _ = fire.Fire(imop.apply_projection_cmd)

