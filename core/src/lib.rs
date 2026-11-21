//! ome_zarr_pyramid in Rust, shipped as the package `ome-zarr-pyramid-core` (Python module
//! `ome_zarr_pyramid_core`). The work lives in the crate `ome-zarr-pyramid`, which knows
//! nothing about Python; this crate only binds it, so ozp's tests can compare both on the
//! same files.

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use ome_zarr_pyramid as ozp;

create_exception!(ome_zarr_pyramid_core, OmeError, PyValueError);

fn err(e: ozp::OmeError) -> PyErr {
    OmeError::new_err(e.message)
}

fn json(image: ozp::Image, arrays: bool) -> PyResult<String> {
    let v = image.to_json(arrays).map_err(err)?;
    serde_json::to_string(&v).map_err(|e| OmeError::new_err(e.to_string()))
}

/// The OME-Zarr image in the local folder `path`, as JSON text: version, multiscales
/// (axes, the 0.6 coordinate system, levels with their index-to-world matrix rows, other
/// systems, edges), omero and image-label as read. With `arrays`, each level also
/// carries its array's shape, data type and chunk shape. The GIL is released.
#[pyfunction]
#[pyo3(signature = (path, arrays = true))]
fn read_image(py: Python<'_>, path: PathBuf, arrays: bool) -> PyResult<String> {
    py.detach(|| {
        ozp::Image::open(&path)
            .map_err(err)
            .and_then(|im| json(im, arrays))
    })
}

/// As [`read_image`], for the image at `inner` ("" for the root) in the zip file `zip`.
#[pyfunction]
#[pyo3(signature = (zip, inner = "", arrays = true))]
fn read_image_zip(py: Python<'_>, zip: PathBuf, inner: &str, arrays: bool) -> PyResult<String> {
    let inner = inner.to_string();
    py.detach(|| {
        ozp::Image::open_zip(&zip, &inner)
            .map_err(err)
            .and_then(|im| json(im, arrays))
    })
}

#[pymodule]
fn ome_zarr_pyramid_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("OZP_VERSION", ozp::VERSION)?;
    m.add("READ_VERSIONS", ozp::READ_VERSIONS.to_vec())?;
    m.add("OmeError", m.py().get_type::<OmeError>())?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_zip, m)?)?;
    Ok(())
}
