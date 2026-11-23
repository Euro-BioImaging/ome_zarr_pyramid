//! ome_zarr_pyramid (ozp) in Rust, starting with what WP15's task DR9 needs: an image's
//! multiscales and their coordinate transformations, each level placed in world space as
//! one matrix from array index to world coordinates, and its levels opened as dyna-zarr
//! engine arrays. micro-viewer reads its data through this crate. Labels, plates and
//! collections come next; writing later.
//!
//! The rules follow ome_zarr_pyramid (ozp), so both read the same files the same way:
//!
//! - The version is declared, never derived from the Zarr format: `ome.version` (0.5,
//!   0.6; `ome` without a version reads as 0.4) or `multiscales[0].version` (0.4).
//!   Read: 0.4, 0.5 and 0.6; any other is an error naming it.
//! - 0.4 / 0.5: a level's own transformations apply first, then the multiscale's
//!   (`world = s_ms * (s_ds * i + t_ds) + t_ms` for scales and translations).
//! - 0.6: the image's own coordinate system is the output of the first level's
//!   transformation; every level maps into it; its axes are that system's axes. Where
//!   ozp reads only scale, identity and a [scale, translation] sequence and refuses the
//!   rest, this layer also places affine, rotation and mapAxis levels (the transforms
//!   crate's matrices). The multiscale's own transformations into other systems are
//!   kept as read (`edges`) with the other systems.
//! - No half-pixel shift for any version: the numbers are used as stored.

mod image;

pub use image::{Axis, CoordinateSystem, Image, Level, Location, Multiscale};

/// This crate's version, for the bindings' build information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The versions read.
pub const READ_VERSIONS: [&str; 3] = ["0.4", "0.5", "0.6"];

/// An image cannot be read as asked; the message says why.
#[derive(Debug, Clone, PartialEq)]
pub struct OmeError {
    pub message: String,
}

impl OmeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for OmeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OmeError {}
