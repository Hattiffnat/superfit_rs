use anyhow::{Context, Result, bail};
use std::{env, path::PathBuf};
use superfit_rs::{MeshToAssemblyOptions, mesh_to_assembly};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut opts = MeshToAssemblyOptions::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" | "-i" => {
                input = Some(PathBuf::from(
                    args.next().context("--input requires a path")?,
                ));
            }
            "--output" | "-o" => {
                output = Some(PathBuf::from(
                    args.next().context("--output requires a directory")?,
                ));
            }
            "--grid-res" => {
                opts.grid_res = args
                    .next()
                    .context("--grid-res requires an integer")?
                    .parse()
                    .context("invalid --grid-res")?;
            }
            "--decomp-res" => {
                opts.decomp_res = Some(
                    args.next()
                        .context("--decomp-res requires an integer")?
                        .parse()
                        .context("invalid --decomp-res")?,
                );
            }
            "--decomp-concavity" => {
                opts.decomp_concavity = args
                    .next()
                    .context("--decomp-concavity requires a number")?
                    .parse()
                    .context("invalid --decomp-concavity")?;
            }
            "--max-parts" => {
                opts.max_parts = args
                    .next()
                    .context("--max-parts requires an integer")?
                    .parse()
                    .context("invalid --max-parts")?;
            }
            "--padding" => {
                opts.padding = args
                    .next()
                    .context("--padding requires a number")?
                    .parse()
                    .context("invalid --padding")?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: mesh_to_assembly_rs --input mesh.obj --output DIR [--grid-res N] [--decomp-res N] [--decomp-concavity X] [--max-parts N] [--padding N]\n  --grid-res N          SDF grid resolution (default: 64)\n  --decomp-res N        VHACD voxel resolution (default: min(grid-res, 128))\n  --decomp-concavity X  VHACD concavity threshold; lower values usually produce more parts (default: 0.002)\n  --max-parts N         Upper bound on VHACD convex parts (default: 128)"
                );
                return Ok(());
            }
            _ if input.is_none() => input = Some(PathBuf::from(arg)),
            _ if output.is_none() => output = Some(PathBuf::from(arg)),
            _ => bail!("unexpected argument {arg:?}; use --help for usage"),
        }
    }

    let input = input.context("missing input OBJ; use --help for usage")?;
    let output = output.context("missing output directory; use --help for usage")?;
    println!("Processing {} -> {}", input.display(), output.display());
    mesh_to_assembly(&input, &output, opts)?;
    println!("Wrote assembly and convex parts to {}", output.display());
    Ok(())
}
