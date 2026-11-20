# Superfit Rust

A Rust rewrite/approximation of [github.com/BardOfCodes/superfit](https://github.com/BardOfCodes/superfit/blob/master/scripts/mesh_to_assembly.py). It takes a watertight OBJ mesh, decomposes it into convex parts, fits an ellipsoid to each part, and computes a signed-distance field (SDF).

> This is not a bit-for-bit port of SuperFit's fitting pipeline. It currently uses Parry VHACD and fits **axis-aligned ellipsoids**, so the resulting assembly is an approximation of the input mesh.

## Build

Requires a Rust toolchain compatible with the edition in `Cargo.toml` (Rust 2024 edition).

```sh
cargo build --release
```

## Run

```sh
cargo run --release -- \
  --input test_data/big_pawn_3.obj \
  --output output \
  --grid-res 64 \
  --decomp-res 128
```

Positional input/output paths are also accepted:

```sh
cargo run --release -- test_data/big_pawn_3.obj output
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--input`, `-i` | required | Input OBJ mesh. |
| `--output`, `-o` | required | Directory for generated files. Created if needed. |
| `--grid-res N` | `64` | Resolution along each axis of the cubic SDF grid. A value of 64 creates 64³ samples. This affects `sdf.bin`, **not** the decomposition or assembly detail. |
| `--decomp-res N` | `min(grid-res, 128)` | VHACD voxel resolution. This controls voxelization detail, **not a target part count**; increasing it can improve the decomposition but does not necessarily create more parts. |
| `--decomp-concavity X` | `0.002` | VHACD concavity threshold in `[0, 1]`. Lower values encourage more splits and tighter convex parts, at higher time and memory cost. |
| `--max-parts N` | `128` | Maximum number of VHACD convex parts. This is a cap, not a requested count; VHACD may stop earlier when its concavity criterion is met. |
| `--padding N` | `0.01` | Extra SDF bounding-box padding, as a fraction of each mesh-axis extent. Must be finite and nonnegative. |
| `--help`, `-h` | — | Show command-line usage. |

For example, lower the concavity threshold and raise the cap to allow VHACD to split more finely, independently of SDF size:

```sh
cargo run --release -- --input mesh.obj --output output \
  --grid-res 64 --decomp-res 128 --decomp-concavity 0.001 --max-parts 128
```

The part count is determined by the mesh and VHACD's stopping criteria; these options permit more parts but do not guarantee a particular count. The program reports how many convex parts were actually produced. `--grid-res` only controls SDF density. Higher SDF resolutions increase memory and computation roughly with the cube of the resolution. Higher `--decomp-res`, lower `--decomp-concavity`, or a larger `--max-parts` can make VHACD more expensive; adjust them incrementally.

## Outputs

The output directory contains:

- `assembly.json` — assembly metadata and fitted ellipsoid centers/radii.
- `part_NNN.obj` — convex hull mesh for each nonempty VHACD part. Existing files matching `part_<digits>.obj` in the output directory are removed before new hulls are written, so they cannot be mistaken for parts from the current run.
- `sdf.bin` — bitcode-encoded SDF data, dimensions, and bounding box. It is a binary file, not an OBJ or human-readable JSON file.

## Compute backends

- **SDF:** GPU compute through `wgpu` is attempted first. The GPU path builds and traverses a BVH for triangle distance and inside/outside tests. If no GPU adapter is available, the workload exceeds GPU limits, or GPU computation fails, the SDF is computed on the CPU using Rayon and Parry's BVH queries.
- **Decomposition and primitive fitting:** Parry VHACD and the current Burn Flex ellipsoid fit run on the CPU. GPU acceleration currently applies to the SDF step only.

For very large or complex meshes, GPU driver limits and device stability still vary. The CPU fallback avoids relying on GPU execution, but high SDF resolutions can still require significant time and memory.

## Input requirements and limitations

- Input is an OBJ mesh with finite vertex coordinates and valid triangular or polygon faces. Polygon faces are triangulated when loaded.
- The mesh must be closed and watertight (each undirected edge must occur exactly twice) for signed-distance generation. Open or non-manifold meshes are rejected.
- VHACD produces convex parts; each part is represented in `assembly.json` by a single axis-aligned ellipsoid. This representation can smooth over geometry that is present in the convex hull. Increasing decomposition resolution may help, but more expressive primitives or a different fitting model would be needed to capture arbitrary local detail.
- `--grid-res` only changes the SDF sampling density. It does not increase the number or geometric fidelity of assembly parts.
- VHACD concavity determines when decomposition stops; `--max-parts` is a ceiling rather than a goal. A lower concavity value is the primary way to request finer decomposition, but the resulting part count depends on the mesh.

## Checks

```sh
cargo fmt --check
cargo check --all-targets
cargo test
```
