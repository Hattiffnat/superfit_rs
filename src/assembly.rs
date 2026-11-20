use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor};

use parry3d::{
    math::Vector,
    transformation::vhacd::{VHACD, VHACDParameters},
};
use serde::Serialize;
use std::{fs, fs::File, io::Write, path::Path};

use crate::mesh::TriMesh;

#[derive(Clone, Debug, Serialize)]
pub struct Ellipsoid {
    pub center: [f32; 3],
    /// Positive half-axis lengths, aligned with the normalized input's axes.
    pub radii: [f32; 3],
    pub source_convex_part: usize,
}

#[derive(Debug, Serialize)]
pub struct Assembly {
    pub method: &'static str,
    pub primitive: &'static str,
    pub combine_operator: &'static str,
    pub decomposition_resolution: u32,
    pub decomposition_concavity: f32,
    pub max_convex_parts: u32,
    pub parts: Vec<Ellipsoid>,
}

pub fn decompose_and_fit(
    mesh: &TriMesh,
    out_dir: &Path,
    voxel_resolution: u32,
    concavity: f32,
    max_convex_parts: u32,
) -> Result<Assembly> {
    ensure!(!mesh.verts.is_empty(), "cannot decompose an empty mesh");
    ensure!(
        !mesh.tris.is_empty(),
        "cannot decompose a mesh without triangles"
    );
    ensure!(
        concavity.is_finite() && (0.0..=1.0).contains(&concavity),
        "VHACD concavity must be finite and between 0 and 1"
    );
    ensure!(
        max_convex_parts > 0,
        "maximum convex parts must be positive"
    );

    let vertices: Vec<Vector> = mesh
        .verts
        .iter()
        .map(|p| Vector::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let indices: Vec<[u32; 3]> = mesh
        .tris
        .iter()
        .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect();

    let params = VHACDParameters {
        resolution: voxel_resolution,
        concavity,
        max_convex_hulls: max_convex_parts,
        ..VHACDParameters::default()
    };
    let decomposition = VHACD::decompose(&params, &vertices, &indices, false);
    // Use voxel-derived hulls so their surface sampling/detail follows VHACD resolution.
    let hulls = decomposition.compute_convex_hulls(1);
    ensure!(!hulls.is_empty(), "VHACD produced no convex parts");

    remove_old_hull_files(out_dir)?;
    let mut parts = Vec::with_capacity(hulls.len());
    for (part_index, (hull_vertices, hull_faces)) in hulls.iter().enumerate() {
        if hull_vertices.is_empty() {
            continue;
        }
        let points: Vec<[f32; 3]> = hull_vertices.iter().map(|p| [p.x, p.y, p.z]).collect();
        let fit =
            fit_ellipsoid(&points).with_context(|| format!("fit convex part {part_index}"))?;
        save_hull_obj(
            &points,
            hull_faces,
            &out_dir.join(format!("part_{part_index:03}.obj")),
        )?;
        parts.push(Ellipsoid {
            center: fit.0,
            radii: fit.1,
            source_convex_part: part_index,
        });
    }
    ensure!(!parts.is_empty(), "VHACD produced no nonempty convex parts");
    println!("VHACD produced {} convex parts", parts.len());

    Ok(Assembly {
        method: "parry-vhacd + burn-flex-autodiff",
        primitive: "axis-aligned ellipsoid",
        combine_operator: "union",
        decomposition_resolution: voxel_resolution,
        decomposition_concavity: concavity,
        max_convex_parts,
        parts,
    })
}

fn remove_old_hull_files(out_dir: &Path) -> Result<()> {
    for entry in fs::read_dir(out_dir).with_context(|| format!("read {}", out_dir.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(index) = name
            .strip_prefix("part_")
            .and_then(|name| name.strip_suffix(".obj"))
        else {
            continue;
        };
        if !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()) {
            fs::remove_file(&path)
                .with_context(|| format!("remove stale hull {}", path.display()))?;
        }
    }
    Ok(())
}

fn fit_ellipsoid(points: &[[f32; 3]]) -> Result<([f32; 3], [f32; 3])> {
    ensure!(
        points.len() >= 4,
        "a convex part needs at least four vertices"
    );
    let count = points.len();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut center = [0.0f32; 3];
    for p in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
            center[axis] += p[axis] / count as f32;
        }
    }
    let mut radii = [0.0f32; 3];
    for axis in 0..3 {
        radii[axis] = ((max[axis] - min[axis]) * 0.5).max(1e-3);
    }

    // Fit the implicit ellipsoid surface: ||(p - center) / radii|| ~= 1.
    // Tensor gradients update center and radii without requiring a host optimizer.
    let device: Device = Device::flex().autodiff();
    let data: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();
    let targets = Tensor::<2>::from_data(burn::tensor::TensorData::new(data, [count, 3]), &device);
    let mut center_tensor =
        Tensor::<2>::from_floats([[center[0], center[1], center[2]]], &device).require_grad();
    let mut radii_tensor =
        Tensor::<2>::from_floats([[radii[0], radii[1], radii[2]]], &device).require_grad();
    let center_prior = Tensor::<2>::from_floats([[center[0], center[1], center[2]]], &device);
    let radii_prior = Tensor::<2>::from_floats([[radii[0], radii[1], radii[2]]], &device);
    let learning_rate = 0.01f32;

    for _ in 0..160 {
        let normalized = (targets.clone() - center_tensor.clone()) / radii_tensor.clone();
        let distance = normalized.powf_scalar(2.0).sum_dim(1).sqrt();
        let surface_loss = (distance - 1.0).powf_scalar(2.0).mean();
        let prior_loss = (center_tensor.clone() - center_prior.clone())
            .powf_scalar(2.0)
            .mean()
            + (radii_tensor.clone() - radii_prior.clone())
                .powf_scalar(2.0)
                .mean();
        let loss = surface_loss + prior_loss * 0.05;
        let gradients = loss.backward();
        let center_grad = center_tensor
            .grad(&gradients)
            .context("Burn did not return ellipsoid center gradients")?
            .into_data()
            .try_into_vec_as::<f32>()?;
        let radii_grad = radii_tensor
            .grad(&gradients)
            .context("Burn did not return ellipsoid radius gradients")?
            .into_data()
            .try_into_vec_as::<f32>()?;
        center_tensor = (center_tensor
            - Tensor::<2>::from_floats(
                [[center_grad[0], center_grad[1], center_grad[2]]],
                &device,
            ) * learning_rate)
            .detach();
        radii_tensor = (radii_tensor
            - Tensor::<2>::from_floats([[radii_grad[0], radii_grad[1], radii_grad[2]]], &device)
                * learning_rate)
            .clamp(1e-3, 2.0)
            .detach();
        center_tensor = center_tensor.require_grad();
        radii_tensor = radii_tensor.require_grad();
    }

    let center = center_tensor.into_data().try_into_vec_as::<f32>()?;
    let radii = radii_tensor.into_data().try_into_vec_as::<f32>()?;
    Ok((
        [center[0], center[1], center[2]],
        [radii[0], radii[1], radii[2]],
    ))
}

fn save_hull_obj(points: &[[f32; 3]], faces: &[[u32; 3]], path: &Path) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    for p in points {
        writeln!(file, "v {:.7} {:.7} {:.7}", p[0], p[1], p[2])?;
    }
    for tri in faces {
        writeln!(file, "f {} {} {}", tri[0] + 1, tri[1] + 1, tri[2] + 1)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_hull_cleanup_only_removes_generated_hull_names() {
        let out_dir = std::env::temp_dir().join(format!("superfit-hulls-{}", std::process::id()));
        fs::create_dir_all(&out_dir).unwrap();
        fs::write(out_dir.join("part_000.obj"), "old hull").unwrap();
        fs::write(out_dir.join("part_12.obj"), "old hull").unwrap();
        fs::write(out_dir.join("part_notes.obj"), "keep").unwrap();
        fs::write(out_dir.join("part_2.txt"), "keep").unwrap();

        remove_old_hull_files(&out_dir).unwrap();

        assert!(!out_dir.join("part_000.obj").exists());
        assert!(!out_dir.join("part_12.obj").exists());
        assert!(out_dir.join("part_notes.obj").exists());
        assert!(out_dir.join("part_2.txt").exists());
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn ellipsoid_fit_returns_finite_positive_radii() {
        let points = vec![
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, -0.25],
            [0.0, 0.0, 0.25],
        ];
        let (center, radii) = fit_ellipsoid(&points).unwrap();
        assert!(center.iter().all(|v| v.is_finite()));
        assert!(radii.iter().all(|v| v.is_finite() && *v > 0.0));
    }
}
