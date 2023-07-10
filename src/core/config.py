
unit_map = {
    't': 'Frame',
    'c': 'Channel',
    'z': 'Slice',
    'y': 'Pixel',
    'x': 'Pixel'
}

type_map = {
    't': 'time',
    'c': 'channel',
    'z': 'space',
    'y': 'space',
    'x': 'space'
}

NotMultiscalesException = Exception('Not applicable as this group is not a single image or label.')
NotImageException = Exception('Not applicable as this group is not an image.')
AxisNotFoundException = Exception('Axis information not provided in the metadata.')

