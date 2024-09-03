import numpy as np
import zarr

def _get_groups_along(shapes, axis: int):
    collection = np.array(shapes)
    axes = np.arange(0, collection.shape[1])
    axmask = axes != axis
    select = collection[:, axmask]
    uqs, index = np.unique(select, axis = 0, return_inverse = True)
    grs = []
    for uq in uqs:
        gr = collection[np.all(select == uq, axis = 1)]
        grs.append(gr)
    return grs

def _concat_along(shapes, axis: int):
    # axis = 1
    # shapes = grs1[2]
    collection = np.array(shapes)
    res = collection[0].copy()
    res[axis] = np.sum(collection[:, axis])
    slcs = []
    aggregated = 0
    for low in collection[:, axis]:
        slcs.append((aggregated, aggregated + low))
        aggregated += low
    return res, slcs


### TODO: bu asagidaki kisimda kaldim. Slice base'leri concat ordera göre birlestirerek bir graph olustur.
shapes = [(4, 10, 5), (4, 10, 5), (4, 3, 8), (4, 8, 8), (4, 9, 8), (3, 20, 13)]

collection = np.array(shapes)

concat_order = (1, 2, 0)
slice_bases = [None] * len(concat_order)
for axis in concat_order:
    # axis = 1
    grs = _get_groups_along(collection, axis = axis)
    new_collection = []
    new_slc_bases = []
    for gr in grs:
        new_shape, slcs = _concat_along(gr, axis)
        new_collection.append(new_shape)
        new_slc_bases.append(slcs)
        slice_bases[axis] = new_slc_bases
    collection = new_collection












