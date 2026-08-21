"""Default configuration values for NGFF metadata and pyramid handling."""

# Default axis order for multi-dimensional images
axis_order = 'tczyx'

# Mapping of axis names to their SI units
unit_map = {
    't': 'second',
    'c': 'Channel',
    'z': 'micrometer',
    'y': 'micrometer',
    'x': 'micrometer'
}

# Default downscaling factors for each axis
scale_factor_map = {
    't': 1,
    'c': 1,
    'z': 2,   # ISOTROPIC in the spatial axes: z is halved like y/x, not kept.
    'y': 2,
    'x': 2
}

# Default scale (pixel size in units) for each axis
scale_map = {
    't': 1,
    'c': 1,
    'z': 1,
    'y': 1,
    'x': 1
}

# Mapping of axis names to their NGFF types
type_map = {
    't': 'time',
    'c': 'channel',
    'z': 'space',
    'y': 'space',
    'x': 'space'
}

# Default temporary directory for intermediate files
TempDir = '/tmp/OME-Zarr'

# Reference path (base resolution level)
refpath = '0'

# Maximum memory for rechunking operations
rechunker_maxmem = '1G'

# Array metadata keys to preserve
array_meta_keys = ['chunks', 'shape', 'compressor', 'dtype', 'dimension_separator']
