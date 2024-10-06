import copy, inspect, itertools, os, zarr, shutil, logging
from attrs import define, field, setters
import numcodecs; numcodecs.blosc.use_threads = False
from joblib import Parallel, delayed, parallel_backend
from dask.distributed import Client, LocalCluster
import dask.distributed as distributed
from dask_jobqueue import SLURMCluster
import numpy as np, dask
from dask.distributed import Lock
from ome_zarr_pyramid.utils.general_utils import *

warnings.simplefilter("ignore", distributed.comm.core.CommClosedError)

# dask.config.set({"distributed.comm.retry.count": 10})
# dask.config.set({"distributed.comm.timeouts.connect": 30})
# dask.config.set({"distributed.worker.memory.terminate": False})
# dask.config.set({"distributed.worker.memory.terminate": False})
# dask.config.set({"distributed.comm.retries.connect": 10})
# dask.config.set({"distributed.comm.timeouts.connect": "30s"})

dask.config.set({
    "distributed.comm.retries.connect": 10,  # Retry connection 10 times
    "distributed.comm.timeouts.connect": "30s",  # Set connection timeout to 30 seconds
    "distributed.worker.memory.terminate": False,  # Prevent workers from terminating on memory errors
    "distributed.worker.reconnect": True,  # Workers will try to reconnect if they lose connection
    "distributed.worker.lifetime.duration": "2h",  # Optionally set a maximum worker lifetime
    "distributed.worker.lifetime.stagger": "10m",  # Workers restart staggered over 10 minutes to prevent all restarting at once
    'distributed.scheduler.worker-ttl': None
})


def is_slurm_available():
    return shutil.which("sbatch") is not None

def _assign_block(dest: zarr.Array,
                  source: zarr.Array,
                  slc_dest: slice,
                  slc_source: slice,
                  function = None,
                  **func_args
                  ):
    if function is None:
        dest[slc_dest] = source[slc_source]
    else:
        dest[slc_dest] = function(source[slc_source], **func_args)
    return dest

def _assign_block_with_lock(dest, source, slc_dest, slc_source, function, lock, **func_args):
    with lock:
        dest = _assign_block(dest, source, slc_dest, slc_source, function, **func_args)
    return dest

def assign_array(dest: zarr.Array, # Is zarr.ProcessSynchronizer not compatible with dask.distributed?
                 source: zarr.Array,
                 crop_indices: tuple = None,
                 insert_at: (list, tuple, np.ndarray) = None,
                 block_size = None,
                 dest_slices = None,
                 source_slices = None,
                 func = None,
                 n_jobs = 8,
                 require_sharedmem = None,
                 slurm_params: dict = None,
                 backend = 'dask',
                 verbose = True,
                 **func_args,
                 ):
    sparams = {
        "cores": 8,  # Number of cores per job
        "memory": "8GB",  # Memory per job
        "nanny": True,
        "walltime": "02:00:00",  # Maximum job runtime
        "processes": 1,  # Number of processes (workers) per job
        # "threads_per_worker": 1,  # One thread per worker to enforce single-threading
        # "job_extra": ["--exclusive"],  # Additional Slurm-specific options
        # "queue": "my_partition",  # The Slurm partition to use
        # "interface": "ib0",  # Specify the network interface (if necessary)
        # "log_directory": "logs",  # Directory to save worker logs (optional)
    }
    if slurm_params is not None:
        for key, value in slurm_params:
            sparams[key] = value

    if crop_indices is None:
        crop_indices = tuple([(0, item) for item in source.shape])
    if insert_at is None:
        insert_at = (0, 0, 0)
    t = np.array(crop_indices)
    maxima = t[:, 1]
    minima = t[:, 0]
    shape = maxima - minima
    if block_size is None:
        block_size = source.chunks
    if source_slices is None:
        source_indices = calculate_slice_bases(
                                                crop_indices = crop_indices,
                                                block_size = block_size,
                                                max_shape = source.shape
                                                )
        source_slices = get_slices_from_slice_bases(source_indices)
    if dest_slices is None:
        insert_minima = np.array(insert_at)
        insert_maxima = np.add(shape, insert_at)
        dest_location = tuple(np.vstack((insert_minima, insert_maxima)).T.tolist())
        dest_indices = calculate_slice_bases(
                                              crop_indices = dest_location,
                                              block_size = block_size,
                                              max_shape = dest.shape
                                              )
        dest_slices = get_slices_from_slice_bases(dest_indices)

    has_synchronizer = False
    if hasattr(dest, 'synchronizer'):
        has_synchronizer = dest.synchronizer is not None

    if n_jobs <= 1:
        backend = 'sequential'

    if backend == 'sequential':
        for slc_source, slc_dest in zip(source_slices, dest_slices):
            dest = _assign_block(dest = dest, source = source, slc_dest = slc_dest, slc_source = slc_source, function = func, **func_args)

    elif backend in ('loky', 'multiprocessing'):
        with parallel_backend(backend):
            with Parallel(
                    verbose=verbose,
                    n_jobs=n_jobs,
                    require=require_sharedmem,
                    prefer='threads'
            ) as parallel:
                _ = parallel(
                    delayed(_assign_block)(
                        dest=dest,
                        source=source,
                        slc_dest=slc_dest,
                        slc_source=slc_source,
                        function=func,
                        **func_args
                    )
                    for slc_source, slc_dest in zip(source_slices, dest_slices)
                )

    else:
        if is_slurm_available():
            print(f"Running with SLURM.")
            with SLURMCluster(**sparams) as cluster:
                cluster.scale(jobs = n_jobs)
                with Client(cluster,
                            heartbeat_interval="10s",
                            timeout="120s",
                            ) as client:
                    with parallel_backend('dask',
                                          wait_for_workers_timeout=600
                                          ):
                        lock = Lock('zarr-write-lock')
                        with Parallel(
                                      verbose = verbose,
                                      n_jobs = n_jobs,
                                      require = require_sharedmem
                                      ) as parallel:
                            _ = parallel(
                                delayed(_assign_block_with_lock)(
                                    dest = dest,
                                    source = source,
                                    slc_dest = slc_dest,
                                    slc_source = slc_source,
                                    function = func,
                                    lock = lock,
                                    **func_args
                                )
                                for slc_source, slc_dest in zip(source_slices, dest_slices)
                            )
        else:
            print(f"Running with local cluster.")
            with LocalCluster(n_workers = n_jobs,
                              processes=True,
                              threads_per_worker=1,
                              nanny=True,
                              memory_limit='8GB',
                              # dashboard_address='127.0.0.1:8787',
                              # worker_dashboard_address='127.0.0.1:0',
                              # host='127.0.0.1'
                              ) as cluster:
                cluster.scale(n_jobs)
                with Client(
                        cluster,
                        heartbeat_interval="10s",
                        timeout="120s",
                        ) as client:
                    with parallel_backend('dask'):
                        lock = Lock('zarr-write-lock')
                        with Parallel(
                                      # n_jobs = n_jobs,
                                      verbose = verbose,
                                      require = require_sharedmem
                                      ) as parallel:
                            _ = parallel(
                                delayed(_assign_block_with_lock)(
                                    dest = dest,
                                    source = source,
                                    slc_dest = slc_dest,
                                    slc_source = slc_source,
                                    function = func,
                                    lock = lock,
                                    **func_args
                                )
                                for slc_source, slc_dest in zip(source_slices, dest_slices)
                            )
    return dest

def basic_assign(dest: zarr.Array, # TODO: add support for dask.distributed and dask_queue
                 source: zarr.Array,
                 dest_slice: tuple = None, # This is destination subset slice TODO: support dict?
                 source_slice: (list, tuple, np.ndarray) = None, # This is source subset slice TODO: support dict?
                 block_size = None,
                 n_jobs = 8,
                 require_sharedmem = None,
                 ):
    if dest_slice is None:
        dest_slice = tuple([0, size] for size in dest.shape)
    if source_slice is None:
        source_slice = tuple([0, size] for size in source.shape)

    assert isinstance(dest_slice, (tuple, list, np.ndarray))
    dslc = np.array(dest_slice)
    assert dslc.ndim == 2
    assert dslc[:, 0].size == dest.ndim

    if np.any(np.greater(dslc[:, 1], dest.shape)):
        raise ValueError(f"Destination slice outside destination array's shape.")
    if np.any(np.less(dslc[:, 0], 0)):
        raise ValueError(f"Destination slice outside destination array's shape.")

    if not np.isscalar(source):
        assert isinstance(source_slice, (tuple, list, np.ndarray))
        sslc = np.array(source_slice)
        assert sslc.ndim == 2
        assert sslc[:, 0].size == source.ndim

        if np.any(np.greater(sslc[:, 1], source.shape)):
            raise ValueError(f"Source slice outside source array's shape.")
        if np.any(np.less(sslc[:, 0], 0)):
            raise ValueError(f"Source slice outside source array's shape.")

    sshape = sslc[:, 1] - sslc[:, 0]
    dshape = dslc[:, 1] - dslc[:, 0]
    assert np.allclose(sshape, dshape), f"The source and destination slices must be equal in shape."

    #######
    if block_size is None:
        try:
            block_size = source.chunks
        except: # if source is a numpy array, it does not have chunks
            block_size = dest.chunks
    source_indices = calculate_slice_bases(
                                            crop_indices = sslc,
                                            block_size = block_size,
                                            max_shape = source.shape
                                            )
    source_slices = get_slices_from_slice_bases(source_indices)

    insert_minima = dslc[:, 0]
    insert_maxima = dslc[:, 1]
    dest_location = tuple(np.vstack((insert_minima, insert_maxima)).T.tolist())
    dest_indices = calculate_slice_bases(
                                          crop_indices = dest_location,
                                          block_size = block_size,
                                          max_shape = dest.shape
                                          )
    dest_slices = get_slices_from_slice_bases(dest_indices)

    has_synchronizer = False
    if hasattr(dest, 'synchronizer'):
        has_synchronizer = dest.synchronizer is not None

    if not isinstance(dest.store, zarr.DirectoryStore) or not has_synchronizer:
        for slc_source, slc_dest in zip(source_slices, dest_slices):
            dest = _assign_block(dest = dest, source = source, slc_dest = slc_dest, slc_source = slc_source)
    else:
        with parallel_backend('loky'): # maybe make selectable
            with Parallel(n_jobs = n_jobs,
                          require = require_sharedmem
                          ) as parallel:
                _ = parallel(
                    delayed(_assign_block)(dest = dest, source = source, slc_dest = slc_dest, slc_source = slc_source)
                    for slc_source, slc_dest in zip(source_slices, dest_slices)
                )
    return dest



#
#
#
# import zarr, os
# from pathlib import Path
# # from ome_zarr_pyramid.core.pyramid import Pyramid
# # # from ome_zarr_pyramid.utils import assignment_utils as asutils
# #
# root = Path("data")
# # pyr = Pyramid().from_zarr(root/'filament.zarr')
# #
# # pyr1 = pyr.copy(root/'filament1.zarr', overwrite = True, paths = ['0'])
# # pyr1.scales
# # np.max(pyr1[0])
# #
# # pyr2 = pyr1.copy(root/'filament2.zarr', overwrite = True)
# # pyr2[0][:] = 0
# # np.max(pyr2[0])
# # syncdir = f"/home/oezdemir/.syncdir/filament1.zarr.sync"
# # zarr.group(pyr2.pyramid_root, synchronizer = zarr.ProcessSynchronizer(syncdir))
#
# arr0 = zarr.open_array(root/'filament1.zarr/0')
# arr1 = zarr.open_array(root/'filament2.zarr/0')
# arr1[:] = 0
# arr1.synchronizer
# np.max(arr0)
# np.max(arr1)
#
# assign_array(dest = arr1,
#              source = arr0,
#              block_size = (10, 100, 100),
#              )
# np.max(arr1)
#
#









