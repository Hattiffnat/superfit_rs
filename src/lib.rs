use anyhow::{Context, Result};

use nalgebra::Point3;

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::mesh::{normalize_to_unit_cube, validate_mesh};

mod assembly;

mod mesh;
mod sdf;

#[derive(Clone, Debug)]
pub struct MeshToAssemblyOptions {
    pub grid_res: usize,
    /// Optional VHACD voxel resolution. Defaults to min(grid_res, 128) to limit cost.
    pub decomp_res: Option<usize>,
    /// Maximum VHACD concavity. Lower values encourage more convex parts.
    pub decomp_concavity: f32,
    /// Upper bound on the number of VHACD convex parts.
    pub max_parts: u32,
    pub padding: f64,
}

impl Default for MeshToAssemblyOptions {
    fn default() -> Self {
        Self {
            grid_res: 64,
            decomp_res: None,
            decomp_concavity: 0.002,
            max_parts: 128,
            padding: 0.01,
        }
    }
}

pub fn mesh_to_assembly<P: AsRef<Path>>(
    mesh_path: P,
    out_dir: P,
    opts: MeshToAssemblyOptions,
) -> Result<()> {
    anyhow::ensure!(opts.grid_res >= 2, "grid resolution must be at least 2");
    let decomp_res = opts.decomp_res.unwrap_or(opts.grid_res.min(128));
    anyhow::ensure!(
        decomp_res >= 2,
        "decomposition resolution must be at least 2"
    );
    let vhacd_resolution = u32::try_from(decomp_res)
        .context("decomposition resolution exceeds VHACD's supported integer range")?;
    anyhow::ensure!(
        opts.decomp_concavity.is_finite() && (0.0..=1.0).contains(&opts.decomp_concavity),
        "decomposition concavity must be finite and between 0 and 1"
    );
    anyhow::ensure!(opts.max_parts > 0, "maximum parts must be positive");
    anyhow::ensure!(
        opts.padding.is_finite() && opts.padding >= 0.0,
        "padding must be finite and nonnegative"
    );
    let mut mesh = load_obj(mesh_path.as_ref()).context("load obj")?;
    validate_mesh(&mesh)?;
    normalize_to_unit_cube(&mut mesh, 0.9)?;

    fs::create_dir_all(&out_dir)?;
    let bbox = compute_bbox(&mesh, opts.padding);
    let assembly = assembly::decompose_and_fit(
        &mesh,
        out_dir.as_ref(),
        vhacd_resolution,
        opts.decomp_concavity,
        opts.max_parts,
    )?;
    let assembly_path = out_dir.as_ref().join("assembly.json");
    let assembly_file = File::create(&assembly_path)
        .with_context(|| format!("create {}", assembly_path.display()))?;
    serde_json::to_writer_pretty(assembly_file, &assembly).context("write assembly json")?;

    let (sdf, dims) = match sdf::compute_gpu(&mesh, &bbox, opts.grid_res) {
        Ok(Some(result)) => {
            println!("SDF grid computed on GPU");
            result
        }
        Ok(None) => {
            println!("No GPU adapter available; computing SDF on CPU");
            sdf::compute_cpu(&mesh, &bbox, opts.grid_res)?
        }
        Err(error) => {
            eprintln!("GPU SDF unavailable ({error:#}); falling back to CPU");
            sdf::compute_cpu(&mesh, &bbox, opts.grid_res)?
        }
    };
    // save sdf as binary via bincode
    let sdf_path = out_dir.as_ref().join("sdf.bin");
    let mut f = File::create(&sdf_path)?;

    let bbox_unpacked = (
        [bbox.0.x, bbox.0.y, bbox.0.z],
        [bbox.1.x, bbox.1.y, bbox.1.z],
    );
    let encoded = bitcode::encode(&(sdf, dims, bbox_unpacked));
    f.write_all(&encoded)?;
    // bincode::serialize_into(&mut f, &(&sdf, &dims, &bbox)).context("serialize sdf")?;

    Ok(())
}

fn load_obj(path: &Path) -> Result<mesh::TriMesh> {
    let f = File::open(path)?;
    let br = BufReader::new(f);
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    for (line_no, line) in br.lines().enumerate() {
        let l = line?;
        let s = l.trim();
        if s.starts_with('v') && s.chars().nth(1).map(|c| c.is_whitespace()).unwrap_or(false) {
            let parts: Vec<_> = s.split_whitespace().collect();
            if parts.len() < 4 {
                anyhow::bail!(
                    "OBJ line {}: vertex requires three coordinates",
                    line_no + 1
                );
            }
            let x: f64 = parts[1]
                .parse()
                .with_context(|| format!("OBJ line {}: invalid x coordinate", line_no + 1))?;
            let y: f64 = parts[2]
                .parse()
                .with_context(|| format!("OBJ line {}: invalid y coordinate", line_no + 1))?;
            let z: f64 = parts[3]
                .parse()
                .with_context(|| format!("OBJ line {}: invalid z coordinate", line_no + 1))?;
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                anyhow::bail!(
                    "OBJ line {}: vertex coordinates must be finite",
                    line_no + 1
                );
            }
            verts.push(Point3::new(x, y, z));
        } else if s.starts_with('f') && s.chars().nth(1).map(|c| c.is_whitespace()).unwrap_or(false)
        {
            let parts: Vec<_> = s.split_whitespace().collect();
            if parts.len() < 4 {
                anyhow::bail!(
                    "OBJ line {}: face requires at least three vertices",
                    line_no + 1
                );
            }
            let mut face = Vec::with_capacity(parts.len() - 1);
            for token in &parts[1..] {
                let raw: isize =
                    token
                        .split('/')
                        .next()
                        .unwrap_or("")
                        .parse()
                        .with_context(|| {
                            format!("OBJ line {}: invalid face index {token:?}", line_no + 1)
                        })?;
                anyhow::ensure!(
                    raw != 0,
                    "OBJ line {}: face indices are 1-based",
                    line_no + 1
                );
                let index = if raw < 0 {
                    verts.len() as isize + raw
                } else {
                    raw - 1
                };
                anyhow::ensure!(
                    index >= 0 && (index as usize) < verts.len(),
                    "OBJ line {}: face index {raw} is out of bounds",
                    line_no + 1
                );
                face.push(index as usize);
            }
            for i in 1..face.len() - 1 {
                tris.push([face[0], face[i], face[i + 1]]);
            }
        }
    }
    Ok(mesh::TriMesh { verts, tris })
}

fn compute_bbox(mesh: &mesh::TriMesh, padding: f64) -> (Point3<f64>, Point3<f64>) {
    let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for v in &mesh.verts {
        min.coords = min.coords.inf(&v.coords);
        max.coords = max.coords.sup(&v.coords);
    }
    let diag = max - min;
    let pad = diag * padding;
    (min - pad, max + pad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_faces_are_triangulated_and_negative_indices_resolve() {
        let path = std::env::temp_dir().join(format!("superfit-{}.obj", std::process::id()));
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n").unwrap();
        let mesh = load_obj(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(mesh.tris, vec![[0, 1, 2], [0, 2, 3]]);
    }

    #[test]
    fn invalid_obj_face_index_is_rejected() {
        let path =
            std::env::temp_dir().join(format!("superfit-invalid-{}.obj", std::process::id()));
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n").unwrap();
        assert!(load_obj(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
