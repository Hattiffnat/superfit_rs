use anyhow::{Result, ensure};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct TriMesh {
    pub verts: Vec<Point3<f64>>,
    pub tris: Vec<[usize; 3]>,
}

pub fn validate_mesh(mesh: &TriMesh) -> Result<()> {
    ensure!(!mesh.verts.is_empty(), "mesh has no vertices");
    ensure!(!mesh.tris.is_empty(), "mesh has no triangles");
    let mut edge_counts = HashMap::new();
    for (index, triangle) in mesh.tris.iter().enumerate() {
        ensure!(
            triangle.iter().all(|&vertex| vertex < mesh.verts.len()),
            "triangle {index} references a missing vertex"
        );
        ensure!(
            triangle[0] != triangle[1] && triangle[1] != triangle[2] && triangle[2] != triangle[0],
            "triangle {index} repeats a vertex"
        );
        let a = mesh.verts[triangle[0]];
        let b = mesh.verts[triangle[1]];
        let c = mesh.verts[triangle[2]];
        ensure!(
            (b - a).cross(&(c - a)).norm() > f64::EPSILON,
            "triangle {index} is degenerate"
        );
        for (u, v) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let edge = (u.min(v), u.max(v));
            *edge_counts.entry(edge).or_insert(0usize) += 1;
        }
    }
    ensure!(
        edge_counts.values().all(|&count| count == 2),
        "signed-distance generation requires a watertight two-manifold mesh"
    );
    Ok(())
}

pub fn normalize_to_unit_cube(mesh: &mut TriMesh, margin: f64) -> Result<()> {
    ensure!(!mesh.verts.is_empty(), "mesh has no vertices");
    ensure!(
        margin.is_finite() && margin > 0.0,
        "normalization margin must be positive and finite"
    );

    let mut bmin = mesh.verts[0];
    let mut bmax = mesh.verts[0];

    for v in &mesh.verts {
        bmin.x = bmin.x.min(v.x);
        bmin.y = bmin.y.min(v.y);
        bmin.z = bmin.z.min(v.z);

        bmax.x = bmax.x.max(v.x);
        bmax.y = bmax.y.max(v.y);
        bmax.z = bmax.z.max(v.z);
    }

    let center = Point3::new(
        0.5 * (bmin.x + bmax.x),
        0.5 * (bmin.y + bmax.y),
        0.5 * (bmin.z + bmax.z),
    );

    let extent = Vector3::new(bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z);

    let max_extent = extent.x.max(extent.y).max(extent.z);

    ensure!(
        max_extent.is_finite() && max_extent > f64::EPSILON,
        "mesh has zero or invalid extent"
    );
    let scale = (2.0 * margin) / max_extent;

    for v in &mut mesh.verts {
        *v = Point3::from((v.coords - center.coords) * scale);
    }
    Ok(())
}
