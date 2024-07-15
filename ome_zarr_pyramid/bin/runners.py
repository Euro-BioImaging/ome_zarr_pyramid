import fire
from ome_zarr_pyramid import BasicOperations, Filters, Threshold, Label, Aggregative

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


def operations():
    _ = fire.Fire(BasicOperations)
    return

def filters():
    _ = fire.Fire(Filters)
    return

def threshold():
    _ = fire.Fire(Threshold)
    return

def label():
    _ = fire.Fire(Label)
    return

def aggregative():
    _ = fire.Fire(Aggregative)
    return

