use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Get the directory that the build script is in
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Build the `add.cu` CUDA file
    let status = Command::new("nvcc")
        .current_dir(&Path::new(&dir).join("src"))
        .arg("-ptx")
        .arg("gpu.cu")
        .status()
        .unwrap();

    // Make sure the CUDA file compiled successfully
    if !status.success() {
        panic!("Failed to compile CUDA file");
    }
}
