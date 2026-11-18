use anyhow::{Context, Result, ensure};
use bytemuck::{Pod, Zeroable, cast_slice};
use nalgebra::Point3;
use parry3d::{
    math::Vector as ParryVector,
    query::{PointQuery, Ray, RayCast},
    shape::TriMesh as ParryTriMesh,
};
use rayon::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, mpsc};
use wgpu::util::DeviceExt;

use crate::mesh::TriMesh;

const CHUNK_SIZE: usize = 16_384;
const WORKGROUP_SIZE: u32 = 64;
const BVH_LEAF_TRIANGLES: usize = 8;
const BVH_STACK_SIZE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBvhNode {
    min: [f32; 4],
    max: [f32; 4],
    data: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    min: [f32; 4],
    max: [f32; 4],
    dims: [u32; 4],
    offset_count: [u32; 4],
    node_count: [u32; 4],
}

pub fn compute_cpu(
    mesh: &TriMesh,
    bbox: &(Point3<f64>, Point3<f64>),
    resolution: usize,
) -> Result<(Vec<f32>, (usize, usize, usize))> {
    ensure!(resolution >= 2, "grid resolution must be at least 2");
    ensure!(
        !mesh.tris.is_empty(),
        "cannot compute an SDF without triangles"
    );
    let parry_mesh = to_parry_mesh(mesh)?;
    let (min, max) = bbox;
    let dims = (resolution, resolution, resolution);
    let steps = [
        (max.x - min.x) / (resolution as f64 - 1.0),
        (max.y - min.y) / (resolution as f64 - 1.0),
        (max.z - min.z) / (resolution as f64 - 1.0),
    ];
    let count = resolution
        .checked_mul(resolution)
        .and_then(|n| n.checked_mul(resolution))
        .context("SDF grid size overflow")?;
    let values = (0..count)
        .into_par_iter()
        .map(|index| {
            let ix = index % resolution;
            let iy = (index / resolution) % resolution;
            let iz = index / (resolution * resolution);
            let point = Point3::new(
                min.x + ix as f64 * steps[0],
                min.y + iy as f64 * steps[1],
                min.z + iz as f64 * steps[2],
            );
            sample_sdf_accelerated(&parry_mesh, &point)
        })
        .collect();
    Ok((values, dims))
}

pub fn compute_gpu(
    mesh: &TriMesh,
    bbox: &(Point3<f64>, Point3<f64>),
    resolution: usize,
) -> Result<Option<(Vec<f32>, (usize, usize, usize))>> {
    match catch_unwind(AssertUnwindSafe(|| {
        compute_gpu_inner(mesh, bbox, resolution)
    })) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown wgpu panic");
            anyhow::bail!("GPU SDF backend panicked: {message}");
        }
    }
}

fn compute_gpu_inner(
    mesh: &TriMesh,
    bbox: &(Point3<f64>, Point3<f64>),
    resolution: usize,
) -> Result<Option<(Vec<f32>, (usize, usize, usize))>> {
    ensure!(resolution >= 2, "grid resolution must be at least 2");
    ensure!(
        !mesh.tris.is_empty(),
        "cannot compute an SDF without triangles"
    );
    let count = resolution
        .checked_mul(resolution)
        .and_then(|n| n.checked_mul(resolution))
        .context("SDF grid size overflow")?;
    let resolution_u32 = u32::try_from(resolution).context("grid resolution exceeds GPU range")?;
    ensure!(
        count <= u32::MAX as usize,
        "SDF grid exceeds GPU index range"
    );
    ensure!(
        mesh.verts.len() <= u32::MAX as usize,
        "too many vertices for GPU SDF"
    );
    let triangle_count = mesh.tris.len();
    let triangle_count_u32 =
        u32::try_from(triangle_count).context("too many triangles for GPU SDF")?;
    let (bvh_nodes, bvh_triangles) = build_gpu_bvh(mesh)?;
    let adapter = match pollster::block_on(wgpu::Instance::default().request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        },
    )) {
        Ok(adapter) => adapter,
        Err(_) => return Ok(None),
    };

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("superfit SDF compute device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .context("request wgpu compute device")?;
    let gpu_errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let uncaptured_errors = Arc::clone(&gpu_errors);
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
        if let Ok(mut errors) = uncaptured_errors.lock() {
            errors.push(error.to_string());
        }
    }));
    let lost_errors = Arc::clone(&gpu_errors);
    device.set_device_lost_callback(move |reason, message| {
        if let Ok(mut errors) = lost_errors.lock() {
            errors.push(format!("device lost ({reason:?}): {message}"));
        }
    });

    let limits = device.limits();
    let vertex_bytes = mesh
        .verts
        .len()
        .checked_mul(16)
        .context("vertex buffer size overflow")?;
    let node_bytes = bvh_nodes
        .len()
        .checked_mul(std::mem::size_of::<GpuBvhNode>())
        .context("BVH node buffer size overflow")?;
    let triangle_data_len = bvh_triangles
        .len()
        .checked_add(
            mesh.tris
                .len()
                .checked_mul(3)
                .context("triangle index count overflow")?,
        )
        .context("triangle index count overflow")?;
    let triangle_data_bytes = triangle_data_len
        .checked_mul(std::mem::size_of::<u32>())
        .context("triangle data buffer size overflow")?;
    ensure!(
        vertex_bytes as u64 <= limits.max_storage_buffer_binding_size as u64
            && vertex_bytes as u64 <= limits.max_buffer_size,
        "mesh vertex buffer exceeds GPU buffer limits"
    );
    ensure!(
        node_bytes as u64 <= limits.max_storage_buffer_binding_size as u64
            && node_bytes as u64 <= limits.max_buffer_size,
        "BVH node buffer exceeds GPU buffer limits"
    );
    ensure!(
        triangle_data_bytes as u64 <= limits.max_storage_buffer_binding_size as u64
            && triangle_data_bytes as u64 <= limits.max_buffer_size,
        "BVH and mesh triangle-index buffer exceeds GPU buffer limits"
    );
    ensure!(
        CHUNK_SIZE * 4 <= limits.max_storage_buffer_binding_size as usize
            && (CHUNK_SIZE * 4) as u64 <= limits.max_buffer_size,
        "GPU buffer limits are too small for SDF chunks"
    );

    let points: Vec<[f32; 4]> = mesh
        .verts
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0])
        .collect();
    let bvh_triangle_count = bvh_triangles.len();
    let mut triangle_data = bvh_triangles;
    triangle_data.extend(mesh.tris.iter().flat_map(|tri| {
        tri.iter()
            .map(|&index| u32::try_from(index).unwrap_or(u32::MAX))
    }));
    ensure!(
        triangle_data.iter().all(|&index| index != u32::MAX),
        "mesh vertex index exceeds GPU range"
    );
    let bvh_nodes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SDF BVH nodes"),
        contents: cast_slice(&bvh_nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bvh_triangles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SDF BVH and mesh triangle indices"),
        contents: cast_slice(&triangle_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SDF mesh vertices"),
        contents: cast_slice(&points),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF output chunk"),
        size: (CHUNK_SIZE * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF readback chunk"),
        size: (CHUNK_SIZE * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF parameters"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh signed-distance shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SDF compute bindings"),
        entries: &[
            storage_layout_entry(0, true),
            storage_layout_entry(1, true),
            storage_layout_entry(2, true),
            storage_layout_entry(3, false),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SDF compute layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SDF compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SDF compute resources"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bvh_triangles_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bvh_nodes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let (min, max) = bbox;
    let chunk_size = CHUNK_SIZE
        .min(limits.max_compute_workgroups_per_dimension as usize * WORKGROUP_SIZE as usize);

    let mut values = Vec::with_capacity(count);
    for offset in (0..count).step_by(chunk_size) {
        let chunk_count = (count - offset).min(chunk_size);
        let params = Params {
            min: [min.x as f32, min.y as f32, min.z as f32, 0.0],
            max: [max.x as f32, max.y as f32, max.z as f32, 0.0],
            dims: [
                resolution_u32,
                resolution_u32,
                resolution_u32,
                triangle_count_u32,
            ],
            offset_count: [
                u32::try_from(offset).context("SDF offset exceeds GPU range")?,
                u32::try_from(chunk_count).context("SDF chunk size exceeds GPU range")?,
                0,
                0,
            ],
            node_count: [
                u32::try_from(bvh_nodes.len()).context("GPU BVH has too many nodes")?,
                u32::try_from(bvh_triangle_count).context("GPU BVH has too many triangles")?,
                0,
                0,
            ],
        };
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SDF compute commands"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SDF grid chunk"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((chunk_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &readback_buffer,
            0,
            (chunk_count * std::mem::size_of::<f32>()) as u64,
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback_buffer.slice(0..(chunk_count * 4) as u64);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("wait for GPU SDF work")?;
        if let Ok(errors) = gpu_errors.lock() {
            if !errors.is_empty() {
                anyhow::bail!("GPU SDF device error: {}", errors.join("; "));
            }
        }
        let map_result = receiver
            .recv()
            .context("GPU readback callback was dropped")?;
        map_result.context("map GPU SDF readback")?;
        let mapped = slice
            .get_mapped_range()
            .context("access mapped GPU SDF readback")?;
        values.extend_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        readback_buffer.unmap();
    }
    if let Ok(errors) = gpu_errors.lock() {
        if !errors.is_empty() {
            anyhow::bail!("GPU SDF device error: {}", errors.join("; "));
        }
    }
    Ok(Some((values, (resolution, resolution, resolution))))
}

#[derive(Clone, Copy)]
struct BuildPrimitive {
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    centroid: [f32; 3],
    triangle: u32,
}

fn build_gpu_bvh(mesh: &TriMesh) -> Result<(Vec<GpuBvhNode>, Vec<u32>)> {
    ensure!(
        !mesh.tris.is_empty(),
        "cannot build a BVH without triangles"
    );
    let mut primitives = Vec::with_capacity(mesh.tris.len());
    for (triangle_index, triangle) in mesh.tris.iter().enumerate() {
        let triangle_index = u32::try_from(triangle_index).context("too many BVH primitives")?;
        let a = mesh.verts[triangle[0]];
        let b = mesh.verts[triangle[1]];
        let c = mesh.verts[triangle[2]];
        let bounds_min = [
            a.x.min(b.x).min(c.x) as f32,
            a.y.min(b.y).min(c.y) as f32,
            a.z.min(b.z).min(c.z) as f32,
        ];
        let bounds_max = [
            a.x.max(b.x).max(c.x) as f32,
            a.y.max(b.y).max(c.y) as f32,
            a.z.max(b.z).max(c.z) as f32,
        ];
        let centroid = [
            0.5 * (bounds_min[0] + bounds_max[0]),
            0.5 * (bounds_min[1] + bounds_max[1]),
            0.5 * (bounds_min[2] + bounds_max[2]),
        ];
        primitives.push(BuildPrimitive {
            bounds_min,
            bounds_max,
            centroid,
            triangle: triangle_index,
        });
    }

    let mut nodes = Vec::with_capacity(mesh.tris.len().saturating_mul(2));
    let mut triangle_indices = Vec::with_capacity(mesh.tris.len());
    build_gpu_bvh_node(&mut primitives, &mut nodes, &mut triangle_indices, 0)?;
    Ok((nodes, triangle_indices))
}

fn build_gpu_bvh_node(
    primitives: &mut [BuildPrimitive],
    nodes: &mut Vec<GpuBvhNode>,
    triangle_indices: &mut Vec<u32>,
    depth: usize,
) -> Result<u32> {
    ensure!(!primitives.is_empty(), "cannot build an empty BVH node");
    ensure!(
        depth < BVH_STACK_SIZE,
        "GPU BVH depth exceeds traversal stack"
    );
    let node_index = u32::try_from(nodes.len()).context("GPU BVH node index overflow")?;
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    let mut centroid_min = [f32::INFINITY; 3];
    let mut centroid_max = [f32::NEG_INFINITY; 3];
    for primitive in primitives.iter() {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(primitive.bounds_min[axis]);
            bounds_max[axis] = bounds_max[axis].max(primitive.bounds_max[axis]);
            centroid_min[axis] = centroid_min[axis].min(primitive.centroid[axis]);
            centroid_max[axis] = centroid_max[axis].max(primitive.centroid[axis]);
        }
    }
    nodes.push(GpuBvhNode {
        min: [bounds_min[0], bounds_min[1], bounds_min[2], 0.0],
        max: [bounds_max[0], bounds_max[1], bounds_max[2], 0.0],
        data: [0; 4],
    });

    if primitives.len() <= BVH_LEAF_TRIANGLES {
        let first = u32::try_from(triangle_indices.len()).context("GPU BVH leaf index overflow")?;
        triangle_indices.extend(primitives.iter().map(|primitive| primitive.triangle));
        nodes[node_index as usize].data = [first, 0, primitives.len() as u32, 0];
        return Ok(node_index);
    }

    let axis = (0..3)
        .max_by(|&a, &b| {
            (centroid_max[a] - centroid_min[a]).total_cmp(&(centroid_max[b] - centroid_min[b]))
        })
        .unwrap_or(0);
    let middle = primitives.len() / 2;
    primitives.select_nth_unstable_by(middle, |a, b| a.centroid[axis].total_cmp(&b.centroid[axis]));
    let (left_primitives, right_primitives) = primitives.split_at_mut(middle);
    let left = build_gpu_bvh_node(left_primitives, nodes, triangle_indices, depth + 1)?;
    let right = build_gpu_bvh_node(right_primitives, nodes, triangle_indices, depth + 1)?;
    nodes[node_index as usize].data = [left, right, 0, 0];
    Ok(node_index)
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn to_parry_mesh(mesh: &TriMesh) -> Result<ParryTriMesh> {
    let vertices = mesh
        .verts
        .iter()
        .map(|p| ParryVector::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let indices = mesh
        .tris
        .iter()
        .map(|tri| {
            Ok([
                u32::try_from(tri[0]).context("mesh index exceeds Parry range")?,
                u32::try_from(tri[1]).context("mesh index exceeds Parry range")?,
                u32::try_from(tri[2]).context("mesh index exceeds Parry range")?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    ParryTriMesh::new(vertices, indices).context("build BVH for CPU SDF")
}

fn sample_sdf_accelerated(mesh: &ParryTriMesh, point: &Point3<f64>) -> f32 {
    let point = ParryVector::new(point.x as f32, point.y as f32, point.z as f32);
    let projection = mesh.project_local_point(point, false);
    let distance = (projection.point - point).length();
    if mesh_contains_by_ray_parity(mesh, point) {
        -distance
    } else {
        distance
    }
}

fn mesh_contains_by_ray_parity(mesh: &ParryTriMesh, point: ParryVector) -> bool {
    let direction = ParryVector::new(1.0, 0.371_390_67, 0.529_108_76).normalize();
    let mut origin = point;
    let mut hits = 0usize;
    let epsilon = 1e-5;
    loop {
        let ray = Ray::new(origin, direction);
        let Some(time_of_impact) = mesh.cast_local_ray(&ray, f32::MAX, false) else {
            break;
        };
        hits += 1;
        origin += direction * (time_of_impact + epsilon);
    }
    hits % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_mesh() -> TriMesh {
        let verts = vec![
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
        ];
        let tris = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        TriMesh { verts, tris }
    }

    #[test]
    fn gpu_bvh_contains_each_triangle_once_and_stays_balanced() {
        let mesh = cube_mesh();
        let (nodes, triangle_indices) = build_gpu_bvh(&mesh).unwrap();
        assert_eq!(triangle_indices.len(), mesh.tris.len());
        let mut sorted = triangle_indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..mesh.tris.len() as u32).collect::<Vec<_>>());
        assert_eq!(nodes[0].data[2], 0);
        assert!(nodes.len() > 1);
    }

    #[test]
    fn cpu_sdf_marks_inside_negative_and_outside_positive() {
        let mesh = cube_mesh();
        let bbox = (Point3::new(-1.4, -1.3, -1.2), Point3::new(1.4, 1.3, 1.2));
        let (sdf, dims) = compute_cpu(&mesh, &bbox, 4).unwrap();
        assert_eq!(dims, (4, 4, 4));
        assert!(sdf[1 + 4 + 16] < 0.0);
        assert!(sdf[3 + 4 + 16] > 0.0);
    }

    #[test]
    fn gpu_sdf_matches_cpu_when_an_adapter_is_available() {
        let mesh = cube_mesh();
        let bbox = (Point3::new(-1.4, -1.3, -1.2), Point3::new(1.4, 1.3, 1.2));
        let (cpu, _) = compute_cpu(&mesh, &bbox, 4).unwrap();
        let Some((gpu, _)) = compute_gpu(&mesh, &bbox, 4).unwrap() else {
            return;
        };
        for (cpu, gpu) in cpu.iter().zip(gpu) {
            assert!((cpu - gpu).abs() < 1e-4, "CPU {cpu} differs from GPU {gpu}");
        }
    }
}

const SHADER: &str = r#"
struct Params {
    min: vec4<f32>,
    max: vec4<f32>,
    dims: vec4<u32>,
    offset_count: vec4<u32>,
    node_count: vec4<u32>,
};
struct BvhNode {
    min: vec4<f32>,
    max: vec4<f32>,
    data: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> triangle_data: array<u32>;
@group(0) @binding(2) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn point_triangle_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return length(p - a); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return length(p - b); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return length(p - (a + ab * v));
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return length(p - c); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return length(p - (a + ac * w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return length(p - (b + (c - b) * w));
    }
    let inverse = 1.0 / (va + vb + vc);
    let v = vb * inverse;
    let w = vc * inverse;
    return length(p - (a + ab * v + ac * w));
}

fn ray_hit(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> bool {
    let direction = normalize(vec3<f32>(1.0, 0.37139067, 0.52910876));
    let edge1 = b - a;
    let edge2 = c - a;
    let h = cross(direction, edge2);
    let determinant = dot(edge1, h);
    if abs(determinant) < 1e-7 { return false; }
    let inverse = 1.0 / determinant;
    let s = p - a;
    let u = dot(s, h) * inverse;
    if u < 0.0 || u > 1.0 { return false; }
    let q = cross(s, edge1);
    let v = dot(direction, q) * inverse;
    if v < 0.0 || u + v > 1.0 { return false; }
    return dot(edge2, q) * inverse > 1e-7;
}

fn point_aabb_distance_squared(p: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> f32 {
    let delta = max(max(box_min - p, vec3<f32>(0.0)), p - box_max);
    return dot(delta, delta);
}

fn ray_intersects_aabb(origin: vec3<f32>, direction: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
    let inverse = 1.0 / direction;
    let t0 = (box_min - origin) * inverse;
    let t1 = (box_max - origin) * inverse;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return far >= max(near, 1e-6);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let local_index = id.x;
    if local_index >= params.offset_count.y { return; }
    let global_index = params.offset_count.x + local_index;
    let nx = params.dims.x;
    let ny = params.dims.y;
    let ix = global_index % nx;
    let iy = (global_index / nx) % ny;
    let iz = global_index / (nx * ny);
    let denom = vec3<f32>(f32(params.dims.x - 1u), f32(params.dims.y - 1u), f32(params.dims.z - 1u));
    let p = params.min.xyz + (params.max.xyz - params.min.xyz) * vec3<f32>(f32(ix), f32(iy), f32(iz)) / denom;
    let ray_direction = normalize(vec3<f32>(1.0, 0.37139067, 0.52910876));
    var min_distance_squared = 1e30;
    var hits = 0u;
    var stack: array<u32, 64>;
    var stack_size = 1u;
    stack[0] = 0u;
    while stack_size > 0u {
        stack_size -= 1u;
        let node = bvh_nodes[stack[stack_size]];
        let node_min = node.min.xyz;
        let node_max = node.max.xyz;
        if point_aabb_distance_squared(p, node_min, node_max) > min_distance_squared && !ray_intersects_aabb(p, ray_direction, node_min, node_max) {
            continue;
        }
        if node.data.z > 0u {
            for (var leaf_index = 0u; leaf_index < node.data.z; leaf_index += 1u) {
                let triangle_index = triangle_data[node.data.x + leaf_index];
                let triangle_offset = params.node_count.y + triangle_index * 3u;
                let a = vertices[triangle_data[triangle_offset]].xyz;
                let b = vertices[triangle_data[triangle_offset + 1u]].xyz;
                let c = vertices[triangle_data[triangle_offset + 2u]].xyz;
                let distance = point_triangle_distance(p, a, b, c);
                min_distance_squared = min(min_distance_squared, distance * distance);
                if ray_hit(p, a, b, c) { hits += 1u; }
            }
        } else {
            if stack_size + 2u <= 64u {
                stack[stack_size] = node.data.y;
                stack[stack_size + 1u] = node.data.x;
                stack_size += 2u;
            }
        }
    }
    let distance = sqrt(min_distance_squared);
    output_values[local_index] = select(distance, -distance, (hits & 1u) == 1u);
}
"#;
