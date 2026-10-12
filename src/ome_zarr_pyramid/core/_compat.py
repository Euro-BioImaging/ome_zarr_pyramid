"""Renamed parameters: the old names keep working for a while, with a warning."""
import warnings


def renamed(value, old_value, *, new, old, where, default=None, stacklevel=3):
    """The value of a renamed parameter.

    ``value`` was passed as ``new`` (e.g. ``max_threads``), ``old_value`` as its old
    name ``old`` (e.g. ``max_workers``). The old name works, with a FutureWarning
    naming the new one; passing both is an error. Neither: ``default``.
    """
    if old_value is None:
        return default if value is None else value
    if value is not None:
        raise TypeError(f"{where}: {old}= is the old name of {new}=; pass {new}= only")
    warnings.warn(f"{where}: {old}= is deprecated; it is now called {new}= (same "
                  f"meaning)", FutureWarning, stacklevel=stacklevel)
    return old_value


_warned_downscale_processes = False


def downscale_processes(value, use_multiprocessing, *, where, stacklevel=3):
    """``max_downscale_processes`` (None: off, the default), accepting the old boolean
    ``use_multiprocessing`` (True: one process per level, at most one per CPU). Any value
    above 1 logs, once per process, that the option is rarely useful and will be
    deprecated: it replaces the one-pass write, in which threads share level 0 and every
    level comes from one read, with one process per level that reads level 0 again."""
    global _warned_downscale_processes
    if isinstance(value, bool):                     # the old slot, passed by position
        if use_multiprocessing is not None:
            raise TypeError(f"{where}: pass max_downscale_processes= only")
        value, use_multiprocessing = None, value
    if use_multiprocessing is not None:
        if value is not None:
            raise TypeError(f"{where}: use_multiprocessing= is the old form of "
                            f"max_downscale_processes=; pass max_downscale_processes= only")
        warnings.warn(f"{where}: use_multiprocessing= is deprecated; it is now called "
                      f"max_downscale_processes= (N levels written in parallel processes)",
                      FutureWarning, stacklevel=stacklevel)
        if use_multiprocessing:
            import os
            value = os.cpu_count() or 1
    if value is None or int(value) <= 1:
        return None
    if not _warned_downscale_processes:
        _warned_downscale_processes = True
        import logging
        logging.getLogger("ome_zarr_pyramid").warning(
            f"max_downscale_processes={int(value)} writes each pyramid level in its own "
            f"process. It is rarely faster than the default one-pass write (level 0 holds "
            f"most of the data, and every level then reads level 0 again) and will be "
            f"deprecated. Use max_threads to parallelise a write.")
    return int(value)
