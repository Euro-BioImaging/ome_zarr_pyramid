//! An OME-Zarr image: where it is, its version, its multiscales and each level's matrix.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dyna_zarr_arrays::ZarrArray;
use dyna_zarr_transforms::{Matrix, Transformation};
use serde_json::{Map, Value};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ReadableStorage, StoreKey};
use zarrs_zip::ZipStorageAdapter;

use crate::{OmeError, READ_VERSIONS};

/// Where an image (or any Zarr node) is: a local folder, or a path inside a finished zip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    Folder(PathBuf),
    Zip { zip: PathBuf, inner: String },
}

impl Location {
    /// The node at `rel` (a path with `/`) below this one.
    pub fn child(&self, rel: &str) -> Location {
        let rel = rel.trim_matches('/');
        match self {
            Location::Folder(p) => {
                let mut q = p.clone();
                for part in rel.split('/').filter(|s| !s.is_empty()) {
                    q.push(part);
                }
                Location::Folder(q)
            }
            Location::Zip { zip, inner } => {
                let inner = inner.trim_matches('/');
                let joined = match (inner.is_empty(), rel.is_empty()) {
                    (true, _) => rel.to_string(),
                    (false, true) => inner.to_string(),
                    (false, false) => format!("{inner}/{rel}"),
                };
                Location::Zip {
                    zip: zip.clone(),
                    inner: joined,
                }
            }
        }
    }

    /// How errors name it.
    pub fn label(&self) -> String {
        match self {
            Location::Folder(p) => p.display().to_string(),
            Location::Zip { zip, inner } => format!("{}::{inner}", zip.display()),
        }
    }

    /// The array stored here, opened by the arrays crate.
    pub fn open_array(&self) -> Result<ZarrArray, OmeError> {
        match self {
            Location::Folder(p) => ZarrArray::open(p),
            Location::Zip { zip, inner } => ZarrArray::open_zip(zip, inner),
        }
        .map_err(|e| OmeError::new(e.message))
    }

    /// The store and the key prefix of this node in it.
    fn storage(&self) -> Result<(ReadableStorage, String), OmeError> {
        match self {
            Location::Folder(p) => {
                let store = FilesystemStore::new(p)
                    .map_err(|e| OmeError::new(format!("cannot open {}: {e}", p.display())))?;
                Ok((Arc::new(store), String::new()))
            }
            Location::Zip { zip, inner } => {
                let dir = zip
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let name = zip
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| OmeError::new(format!("{}: not a file name", zip.display())))?;
                let store = FilesystemStore::new(dir)
                    .map_err(|e| OmeError::new(format!("cannot open {}: {e}", dir.display())))?;
                let key = StoreKey::new(name)
                    .map_err(|e| OmeError::new(format!("{}: {e}", zip.display())))?;
                let adapter = ZipStorageAdapter::new(Arc::new(store), key).map_err(|e| {
                    OmeError::new(format!(
                        "{} is not a zip zarrs can read: {e}",
                        zip.display()
                    ))
                })?;
                let inner = inner.trim_matches('/');
                let prefix = if inner.is_empty() {
                    String::new()
                } else {
                    format!("{inner}/")
                };
                Ok((Arc::new(adapter), prefix))
            }
        }
    }

    /// The JSON document stored under `name` in this node (`zarr.json`, `.zattrs`), if
    /// any.
    fn json(&self, name: &str) -> Result<Option<Value>, OmeError> {
        let (storage, prefix) = self.storage()?;
        let key = StoreKey::new(format!("{prefix}{name}"))
            .map_err(|e| OmeError::new(format!("{}: {e}", self.label())))?;
        let bytes = storage
            .get(&key)
            .map_err(|e| OmeError::new(format!("{}: reading {name}: {e}", self.label())))?;
        match bytes {
            None => Ok(None),
            Some(b) => serde_json::from_slice(&b)
                .map(Some)
                .map_err(|e| OmeError::new(format!("{}: {name} is not JSON: {e}", self.label()))),
        }
    }

    /// The node's attributes: `zarr.json`'s `attributes` (Zarr v3) or `.zattrs` (v2).
    pub fn attributes(&self) -> Result<Map<String, Value>, OmeError> {
        if let Some(doc) = self.json("zarr.json")? {
            return Ok(doc
                .get("attributes")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default());
        }
        match self.json(".zattrs")? {
            Some(Value::Object(m)) => Ok(m),
            Some(other) => Err(OmeError::new(format!(
                "{}: .zattrs is not an object: {other}",
                self.label()
            ))),
            None => Err(OmeError::new(format!(
                "{}: no zarr.json or .zattrs: not a Zarr group",
                self.label()
            ))),
        }
    }
}

/// An axis as read: its name, its NGFF type and unit when given.
#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    pub name: String,
    pub kind: Option<String>,
    pub unit: Option<String>,
}

/// A coordinate system (NGFF 0.6): its name and axes.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateSystem {
    pub name: String,
    pub axes: Vec<Axis>,
}

/// One resolution level: its array's path below the image, its transformations as read,
/// and the matrix from its array index to world coordinates (homogeneous, over the
/// multiscale's axes).
#[derive(Debug, Clone, PartialEq)]
pub struct Level {
    pub path: String,
    pub transformations: Vec<Transformation>,
    pub matrix: Matrix,
}

/// A multiscale image: its axes and levels; for NGFF 0.6, the coordinate system the
/// levels map into, the other systems and the transformations into them, as read.
#[derive(Debug, Clone, PartialEq)]
pub struct Multiscale {
    pub name: Option<String>,
    pub axes: Vec<Axis>,
    /// NGFF 0.6: the image's own coordinate system (the levels' output).
    pub system: Option<String>,
    pub levels: Vec<Level>,
    /// NGFF 0.6: the other coordinate systems, as read.
    pub systems: Vec<CoordinateSystem>,
    /// NGFF 0.6: the multiscale's transformations into other systems, as read.
    pub edges: Vec<Transformation>,
}

/// An OME-Zarr image.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub location: Location,
    pub version: String,
    pub multiscales: Vec<Multiscale>,
    pub omero: Option<Value>,
    pub image_label: Option<Value>,
}

fn version_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn as_list(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(a) => a.clone(),
        Value::Null => Vec::new(),
        other => vec![other.clone()],
    }
}

fn axis(v: &Value, at: &str) -> Result<Axis, OmeError> {
    match v {
        // a bare name, as some old writers stored axes
        Value::String(s) => Ok(Axis {
            name: s.to_lowercase(),
            kind: None,
            unit: None,
        }),
        Value::Object(m) => {
            let name = m
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| OmeError::new(format!("{at}: an axis without a name: {v}")))?;
            let text = |k: &str| m.get(k).and_then(Value::as_str).map(str::to_string);
            Ok(Axis {
                name: name.to_string(),
                kind: text("type"),
                unit: text("unit"),
            })
        }
        other => Err(OmeError::new(format!("{at}: an axis is {other}"))),
    }
}

fn axes(v: Option<&Value>, at: &str) -> Result<Vec<Axis>, OmeError> {
    let list = v
        .and_then(Value::as_array)
        .ok_or_else(|| OmeError::new(format!("{at}: no axes")))?;
    let out = list
        .iter()
        .map(|a| axis(a, at))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, a) in out.iter().enumerate() {
        if out[..i].iter().any(|b| b.name == a.name) {
            return Err(OmeError::new(format!("{at}: axis {:?} twice", a.name)));
        }
    }
    Ok(out)
}

fn transformations(v: Option<&Value>, at: &str) -> Result<Vec<Transformation>, OmeError> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    as_list(v)
        .iter()
        .map(|t| {
            Transformation::from_json(t).map_err(|e| OmeError::new(format!("{at}: {}", e.message)))
        })
        .collect()
}

/// `ts` applied one after another (the first first), over `ndim` axes.
fn chain(ts: &[Transformation], ndim: usize, at: &str) -> Result<Matrix, OmeError> {
    let mut m = Matrix::identity(ndim);
    for t in ts {
        let step = t
            .matrix(ndim)
            .map_err(|e| OmeError::new(format!("{at}: {}", e.message)))?;
        m = step
            .matmul(&m)
            .map_err(|e| OmeError::new(format!("{at}: {}", e.message)))?;
    }
    Ok(m)
}

/// A transformation end's coordinate system name (`{"name": ...}` or a bare string).
fn end_name(end: Option<&Value>) -> Option<String> {
    match end? {
        Value::String(s) => Some(s.clone()),
        Value::Object(m) => m.get("name").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

fn datasets<'a>(ms: &'a Map<String, Value>, at: &str) -> Result<&'a Vec<Value>, OmeError> {
    let ds = ms
        .get("datasets")
        .and_then(Value::as_array)
        .ok_or_else(|| OmeError::new(format!("{at}: no datasets")))?;
    if ds.is_empty() {
        return Err(OmeError::new(format!("{at}: no datasets")));
    }
    Ok(ds)
}

fn dataset_path(d: &Value, at: &str) -> Result<String, OmeError> {
    d.get("path")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| OmeError::new(format!("{at}: a dataset without a path: {d}")))
}

/// A 0.4 / 0.5 multiscale: each level's transformations, then the multiscale's.
fn multiscale_04(ms: &Map<String, Value>, at: &str) -> Result<Multiscale, OmeError> {
    let axes = axes(ms.get("axes"), at)?;
    let n = axes.len();
    let outer = transformations(ms.get("coordinateTransformations"), at)?;
    let mut levels = Vec::new();
    for d in datasets(ms, at)? {
        let path = dataset_path(d, at)?;
        let here = format!("{at}, level {path:?}");
        let own = transformations(d.get("coordinateTransformations"), &here)?;
        let mut all = own.clone();
        all.extend(outer.iter().cloned());
        let matrix = chain(&all, n, &here)?;
        levels.push(Level {
            path,
            transformations: own,
            matrix,
        });
    }
    Ok(Multiscale {
        name: ms.get("name").and_then(Value::as_str).map(str::to_string),
        axes,
        system: None,
        levels,
        systems: Vec::new(),
        edges: outer,
    })
}

/// A 0.6 multiscale: every level maps into the image's own coordinate system, the
/// output of the first level's transformation.
fn multiscale_06(ms: &Map<String, Value>, at: &str) -> Result<Multiscale, OmeError> {
    let ds = datasets(ms, at)?;
    let systems: Vec<CoordinateSystem> = ms
        .get("coordinateSystems")
        .and_then(Value::as_array)
        .ok_or_else(|| OmeError::new(format!("{at}: no coordinateSystems")))?
        .iter()
        .map(|s| {
            let name = s.get("name").and_then(Value::as_str).ok_or_else(|| {
                OmeError::new(format!("{at}: a coordinate system without a name: {s}"))
            })?;
            Ok(CoordinateSystem {
                name: name.to_string(),
                axes: axes(s.get("axes"), &format!("{at}, system {name:?}"))?,
            })
        })
        .collect::<Result<_, OmeError>>()?;
    let first = transformations(ds[0].get("coordinateTransformations"), at)?;
    let intrinsic = first
        .first()
        .and_then(|t| end_name(t.output.as_ref()))
        .ok_or_else(|| {
            OmeError::new(format!(
                "{at}: the first level's transformation names no output coordinate system"
            ))
        })?;
    let own = systems
        .iter()
        .find(|s| s.name == intrinsic)
        .ok_or_else(|| {
            OmeError::new(format!(
                "{at}: the levels map into {intrinsic:?}, which is not among its coordinateSystems"
            ))
        })?
        .clone();
    let n = own.axes.len();
    let mut levels = Vec::new();
    for d in ds {
        let path = dataset_path(d, at)?;
        let here = format!("{at}, level {path:?}");
        let ts = transformations(d.get("coordinateTransformations"), &here)?;
        if ts.is_empty() {
            return Err(OmeError::new(format!(
                "{here}: no coordinateTransformations"
            )));
        }
        for t in &ts {
            if end_name(t.output.as_ref()).as_deref() != Some(intrinsic.as_str()) {
                return Err(OmeError::new(format!(
                    "{here}: a transformation into {:?}, not the image's system {intrinsic:?}",
                    end_name(t.output.as_ref())
                )));
            }
        }
        let matrix = chain(&ts, n, &here)?;
        levels.push(Level {
            path,
            transformations: ts,
            matrix,
        });
    }
    Ok(Multiscale {
        name: ms.get("name").and_then(Value::as_str).map(str::to_string),
        axes: own.axes.clone(),
        system: Some(intrinsic),
        levels,
        systems: systems.into_iter().filter(|s| s.name != own.name).collect(),
        edges: transformations(ms.get("coordinateTransformations"), at)?,
    })
}

impl Image {
    /// The image in the local folder `path`.
    pub fn open(path: &Path) -> Result<Self, OmeError> {
        Self::at(Location::Folder(path.to_path_buf()))
    }

    /// The image at `inner` ("" for the root) in the zip file `zip`.
    pub fn open_zip(zip: &Path, inner: &str) -> Result<Self, OmeError> {
        Self::at(Location::Zip {
            zip: zip.to_path_buf(),
            inner: inner.to_string(),
        })
    }

    /// The image at `location`.
    pub fn at(location: Location) -> Result<Self, OmeError> {
        let attrs = location.attributes()?;
        Self::from_attributes(location, &attrs)
    }

    /// The image whose group attributes are `attrs` (as stored), at `location`.
    pub fn from_attributes(
        location: Location,
        attrs: &Map<String, Value>,
    ) -> Result<Self, OmeError> {
        let at = location.label();
        // the metadata block: `ome` (0.5, 0.6) or the attributes themselves (0.4)
        let (meta, version) = if let Some(ome) = attrs.get("ome") {
            let ome = ome
                .as_object()
                .ok_or_else(|| OmeError::new(format!("{at}: \"ome\" is not an object")))?;
            if ome.get("type").and_then(Value::as_str).is_some() {
                return Err(OmeError::new(format!(
                    "{at}: an RFC 8 collection node, not an image"
                )));
            }
            let v = ome
                .get("version")
                .and_then(version_string)
                .unwrap_or_else(|| "0.4".to_string());
            (ome.clone(), v)
        } else if let Some(ms) = attrs.get("multiscales") {
            let list = as_list(ms);
            let v = list
                .first()
                .and_then(|m| m.get("version"))
                .and_then(version_string)
                .unwrap_or_else(|| "0.4".to_string());
            (attrs.clone(), v)
        } else {
            return Err(OmeError::new(format!(
                "{at}: no OME-Zarr image metadata (no multiscales)"
            )));
        };
        if !READ_VERSIONS.contains(&version.as_str()) {
            return Err(OmeError::new(format!(
                "{at}: OME-Zarr version {version:?} is not one this layer reads ({})",
                READ_VERSIONS.join(", ")
            )));
        }
        let list = as_list(meta.get("multiscales").unwrap_or(&Value::Null));
        if list.is_empty() {
            return Err(OmeError::new(format!("{at}: no multiscales")));
        }
        let mut multiscales = Vec::new();
        for (i, ms) in list.iter().enumerate() {
            let ms = ms
                .as_object()
                .ok_or_else(|| OmeError::new(format!("{at}: multiscale {i} is not an object")))?;
            let here = format!("{at}, multiscale {i}");
            multiscales.push(if version == "0.6" {
                multiscale_06(ms, &here)?
            } else {
                multiscale_04(ms, &here)?
            });
        }
        let block = |k: &str| meta.get(k).or_else(|| attrs.get(k)).cloned();
        Ok(Self {
            location,
            version,
            multiscales,
            omero: block("omero"),
            image_label: block("image-label"),
        })
    }

    /// Level `level` of multiscale `ms`, opened as an engine array.
    pub fn level_array(&self, ms: usize, level: usize) -> Result<ZarrArray, OmeError> {
        let m = self
            .multiscales
            .get(ms)
            .ok_or_else(|| OmeError::new(format!("no multiscale {ms}")))?;
        let l = m
            .levels
            .get(level)
            .ok_or_else(|| OmeError::new(format!("no level {level} in multiscale {ms}")))?;
        self.location.child(&l.path).open_array()
    }

    /// The image as JSON: version, multiscales (axes, system, levels with their matrix
    /// as NGFF's affine form, n rows of n + 1 with the last row 0 ... 0 1 implied, other
    /// systems, edges), omero and image-label as read. With `arrays`, each
    /// level also carries its array's shape, data type and chunk shape.
    pub fn to_json(&self, arrays: bool) -> Result<Value, OmeError> {
        let axis_json = |a: &Axis| {
            let mut m = Map::new();
            m.insert("name".into(), Value::from(a.name.clone()));
            if let Some(k) = &a.kind {
                m.insert("type".into(), Value::from(k.clone()));
            }
            if let Some(u) = &a.unit {
                m.insert("unit".into(), Value::from(u.clone()));
            }
            Value::Object(m)
        };
        let mut mss = Vec::new();
        for (i, ms) in self.multiscales.iter().enumerate() {
            let mut levels = Vec::new();
            for (j, l) in ms.levels.iter().enumerate() {
                let mut m = Map::new();
                m.insert("path".into(), Value::from(l.path.clone()));
                m.insert("matrix".into(), Value::from(l.matrix.rows()));
                m.insert(
                    "transformations".into(),
                    Value::Array(
                        l.transformations
                            .iter()
                            .map(Transformation::to_json)
                            .collect(),
                    ),
                );
                if arrays {
                    let a = self.level_array(i, j)?;
                    use dyna_zarr_arrays::Source;
                    m.insert("shape".into(), Value::from(a.shape().to_vec()));
                    m.insert("dtype".into(), Value::from(a.dtype().name()));
                    m.insert("chunks".into(), Value::from(a.chunks().to_vec()));
                }
                levels.push(Value::Object(m));
            }
            let mut m = Map::new();
            m.insert(
                "name".into(),
                ms.name.clone().map_or(Value::Null, Value::from),
            );
            m.insert(
                "axes".into(),
                Value::Array(ms.axes.iter().map(axis_json).collect()),
            );
            m.insert(
                "system".into(),
                ms.system.clone().map_or(Value::Null, Value::from),
            );
            m.insert("levels".into(), Value::Array(levels));
            m.insert(
                "systems".into(),
                Value::Array(
                    ms.systems
                        .iter()
                        .map(|s| {
                            let mut o = Map::new();
                            o.insert("name".into(), Value::from(s.name.clone()));
                            o.insert(
                                "axes".into(),
                                Value::Array(s.axes.iter().map(axis_json).collect()),
                            );
                            Value::Object(o)
                        })
                        .collect(),
                ),
            );
            m.insert(
                "edges".into(),
                Value::Array(ms.edges.iter().map(Transformation::to_json).collect()),
            );
            mss.push(Value::Object(m));
        }
        let mut out = Map::new();
        out.insert("version".into(), Value::from(self.version.clone()));
        out.insert("multiscales".into(), Value::Array(mss));
        out.insert("omero".into(), self.omero.clone().unwrap_or(Value::Null));
        out.insert(
            "image_label".into(),
            self.image_label.clone().unwrap_or(Value::Null),
        );
        Ok(Value::Object(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn loc() -> Location {
        Location::Folder(PathBuf::from("test"))
    }

    fn attrs(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    fn world(m: &Matrix, i: &[f64]) -> Vec<f64> {
        m.apply(i)
    }

    #[test]
    fn v04_levels_then_the_multiscale() {
        // ozp's composition test: world = s_ms * (s_ds * i + t_ds) + t_ms
        let a = attrs(json!({"multiscales": [{"version": "0.4", "name": "img",
            "axes": [{"name": "z", "type": "space", "unit": "micrometer"},
                     {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.5, 0.5]}]},
                {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]},
                                                            {"type": "translation", "translation": [0.0, 0.25, 0.25]}]}],
            "coordinateTransformations": [{"type": "identity"}, {"type": "scale", "scale": [1.0, 2.0, 2.0]},
                                          {"type": "translation", "translation": [5.0, 6.0, 7.0]}]}]}));
        let im = Image::from_attributes(loc(), &a).unwrap();
        assert_eq!(im.version, "0.4");
        let ms = &im.multiscales[0];
        assert_eq!(ms.axes[0].unit.as_deref(), Some("micrometer"));
        assert_eq!(
            world(&ms.levels[0].matrix, &[1.0, 2.0, 4.0]),
            vec![6.0, 8.0, 11.0]
        );
        // level 1: s_ms * (1 * i + 0.25) + t_ms
        assert_eq!(
            world(&ms.levels[1].matrix, &[0.0, 1.0, 2.0]),
            vec![5.0, 8.5, 11.5]
        );
    }

    #[test]
    fn v05_under_ome_and_unknown_versions() {
        let ms = json!([{"axes": [{"name": "y"}, {"name": "x"}],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 3.0]}]}]}]);
        let im = Image::from_attributes(
            loc(),
            &attrs(json!({"ome": {"version": "0.5", "multiscales": ms}})),
        )
        .unwrap();
        assert_eq!(im.version, "0.5");
        assert_eq!(
            world(&im.multiscales[0].levels[0].matrix, &[1.0, 1.0]),
            vec![2.0, 3.0]
        );
        // `ome` without a version reads as 0.4 (ozp)
        let im =
            Image::from_attributes(loc(), &attrs(json!({"ome": {"multiscales": ms}}))).unwrap();
        assert_eq!(im.version, "0.4");
        let e = Image::from_attributes(
            loc(),
            &attrs(json!({"ome": {"version": "0.7", "multiscales": ms}})),
        )
        .unwrap_err();
        assert!(e.message.contains("\"0.7\""), "{}", e.message);
        assert!(Image::from_attributes(loc(), &attrs(json!({"foo": 1}))).is_err());
        assert!(Image::from_attributes(
            loc(),
            &attrs(json!({"ome": {"version": "0.x", "type": "collection"}}))
        )
        .is_err());
    }

    #[test]
    fn v06_levels_map_into_the_image_system_with_any_modelled_type() {
        let a = attrs(json!({"ome": {"version": "0.6", "multiscales": [{
            "coordinateSystems": [
                {"name": "physical", "axes": [{"name": "y", "type": "space", "unit": "micrometer"},
                                              {"name": "x", "type": "space", "unit": "micrometer"}]},
                {"name": "stage", "axes": [{"name": "y"}, {"name": "x"}]}],
            "datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "sequence",
                    "transformations": [{"type": "scale", "scale": [0.5, 0.5]},
                                        {"type": "translation", "translation": [10.0, 20.0]}],
                    "input": {"path": "0"}, "output": {"name": "physical"}}]},
                {"path": "1", "coordinateTransformations": [{"type": "affine",
                    "affine": [[0.0, -1.0, 3.0], [1.0, 0.0, 4.0]],
                    "input": {"path": "1"}, "output": {"name": "physical"}}]}],
            "coordinateTransformations": [{"type": "translation", "translation": [100.0, 0.0],
                "input": {"name": "physical"}, "output": {"name": "stage"}}]}]}}));
        let im = Image::from_attributes(loc(), &a).unwrap();
        let ms = &im.multiscales[0];
        assert_eq!(ms.system.as_deref(), Some("physical"));
        assert_eq!(ms.systems.len(), 1);
        assert_eq!(ms.edges.len(), 1);
        assert_eq!(world(&ms.levels[0].matrix, &[2.0, 4.0]), vec![11.0, 22.0]);
        // a rotation by 90 degrees and a shift: (y, x) -> (-x + 3, y + 4)
        assert_eq!(world(&ms.levels[1].matrix, &[1.0, 2.0]), vec![1.0, 5.0]);
    }

    #[test]
    fn v06_refuses_what_it_cannot_place() {
        let base = |second_output: &str| {
            attrs(json!({"ome": {"version": "0.6", "multiscales": [{
            "coordinateSystems": [{"name": "physical", "axes": [{"name": "y"}, {"name": "x"}]}],
            "datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0],
                    "output": {"name": "physical"}}]},
                {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0],
                    "output": {"name": second_output}}]}]}]}}))
        };
        assert!(Image::from_attributes(loc(), &base("physical")).is_ok());
        let e = Image::from_attributes(loc(), &base("other")).unwrap_err();
        assert!(
            e.message.contains("not the image's system"),
            "{}",
            e.message
        );
        let missing = attrs(json!({"ome": {"version": "0.6", "multiscales": [{
            "coordinateSystems": [{"name": "a", "axes": [{"name": "x"}]}],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0],
                "output": {"name": "b"}}]}]}]}}));
        assert!(Image::from_attributes(loc(), &missing)
            .unwrap_err()
            .message
            .contains("not among"));
        let carried = attrs(json!({"ome": {"version": "0.6", "multiscales": [{
            "coordinateSystems": [{"name": "a", "axes": [{"name": "x"}]}],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "displacements",
                "path": "d", "output": {"name": "a"}}]}]}]}}));
        assert!(Image::from_attributes(loc(), &carried).is_err());
    }

    #[test]
    fn child_locations() {
        let f = Location::Folder(PathBuf::from("a"));
        assert_eq!(
            f.child("labels/cells/0"),
            Location::Folder(PathBuf::from("a").join("labels").join("cells").join("0"))
        );
        let z = Location::Zip {
            zip: PathBuf::from("x.zip"),
            inner: "".into(),
        };
        assert_eq!(
            z.child("0"),
            Location::Zip {
                zip: PathBuf::from("x.zip"),
                inner: "0".into()
            }
        );
        let z = Location::Zip {
            zip: PathBuf::from("x.zip"),
            inner: "img/".into(),
        };
        assert_eq!(
            z.child("/0"),
            Location::Zip {
                zip: PathBuf::from("x.zip"),
                inner: "img/0".into()
            }
        );
    }
}
