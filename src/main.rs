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
            "--padding" => {
                opts.padding = args
                    .next()
                    .context("--padding requires a number")?
                    .parse()
                    .context("invalid --padding")?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: mesh_to_assembly_rs --input mesh.obj --output DIR [--grid-res N] [--padding N]"
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
    println!(
        "Wrote assembly to {}",
        output.join("assembly.json").display()
    );
    Ok(())
}
