use anyhow::{Context, Result};

use nalgebra::{Point3, Vector3};

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::mesh::{normalize_to_unit_cube, validate_mesh};

mod assembly;

mod mesh;

#[derive(Clone, Debug)]
pub struct MeshToAssemblyOptions {
    pub grid_res: usize,
    pub padding: f64,
}

impl Default for MeshToAssemblyOptions {
    fn default() -> Self {
        Self {
            grid_res: 64,
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
    anyhow::ensure!(
        opts.padding.is_finite() && opts.padding >= 0.0,
        "padding must be finite and nonnegative"
    );
    let mut mesh = load_obj(mesh_path.as_ref()).context("load obj")?;
    validate_mesh(&mesh)?;
    normalize_to_unit_cube(&mut mesh, 0.9)?;

    fs::create_dir_all(&out_dir)?;
    let bbox = compute_bbox(&mesh, opts.padding);
    let assembly = assembly::decompose_and_fit(&mesh, out_dir.as_ref())?;
    let assembly_path = out_dir.as_ref().join("assembly.json");
    let assembly_file = File::create(&assembly_path)
        .with_context(|| format!("create {}", assembly_path.display()))?;
    serde_json::to_writer_pretty(assembly_file, &assembly).context("write assembly json")?;

    let (sdf, dims) = compute_sdf_grid(&mesh, &bbox, opts.grid_res)?;
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

fn point_triangle_distance(
    p: &Point3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
    c: &Point3<f64>,
) -> f64 {
    // using algorithm by David Eberly / real-time collision detection
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (p - a).norm();
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (p - b).norm();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let proj = a + ab * v;
        return (p - proj).norm();
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (p - c).norm();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let proj = a + ac * w;
        return (p - proj).norm();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let proj = b + (c - b) * w;
        return (p - proj).norm();
    }
    // inside face region
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let proj = a + ab * v + ac * w;
    (p - proj).norm()
}

fn ray_intersects_triangle(
    orig: &Point3<f64>,
    dir: &Vector3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
    c: &Point3<f64>,
) -> Option<f64> {
    // Moller–Trumbore
    let eps = 1e-9;
    let edge1 = b - a;
    let edge2 = c - a;
    let h = dir.cross(&edge2);
    let det = edge1.dot(&h);
    if det.abs() < eps {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = orig - a;
    let u = s.dot(&h) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(&edge1);
    let v = dir.dot(&q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(&q) * inv_det;
    if t > eps { Some(t) } else { None }
}

fn compute_sdf_grid(
    mesh: &mesh::TriMesh,
    bbox: &(Point3<f64>, Point3<f64>),
    grid_res: usize,
) -> Result<(Vec<f32>, (usize, usize, usize))> {
    let (min, max) = bbox;
    let nx = grid_res;
    let ny = grid_res;
    let nz = grid_res;
    let sx = (max.x - min.x) / (nx as f64 - 1.0);
    let sy = (max.y - min.y) / (ny as f64 - 1.0);
    let sz = (max.z - min.z) / (nz as f64 - 1.0);
    let mut sdf = vec![f32::MAX; nx * ny * nz];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iz * ny * nx + iy * nx + ix;
                let p = Point3::new(
                    min.x + ix as f64 * sx,
                    min.y + iy as f64 * sy,
                    min.z + iz as f64 * sz,
                );
                // compute closest distance
                let mut min_d = f64::INFINITY;
                for t in &mesh.tris {
                    let d = point_triangle_distance(
                        &p,
                        &mesh.verts[t[0]],
                        &mesh.verts[t[1]],
                        &mesh.verts[t[2]],
                    );
                    if d < min_d {
                        min_d = d;
                    }
                }
                // sign via raycast along +X
                let dir = Vector3::new(1.0, 0.0, 0.0);
                let mut hits = 0usize;
                for t in &mesh.tris {
                    if let Some(_t) = ray_intersects_triangle(
                        &p,
                        &dir,
                        &mesh.verts[t[0]],
                        &mesh.verts[t[1]],
                        &mesh.verts[t[2]],
                    ) {
                        hits += 1;
                    }
                }
                let sign = if hits % 2 == 1 { -1.0 } else { 1.0 };
                sdf[idx] = (min_d * sign) as f32;
            }
        }
    }

    Ok((sdf, (nx, ny, nz)))
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
