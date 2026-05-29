use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let cpp_dir = manifest_dir.join("../../cpp");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let build_dir = out_dir.join("cmake-build");

    run(Command::new("cmake")
        .arg("-S")
        .arg(&cpp_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-G")
        .arg("Ninja")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_PREFIX_PATH={}", torch_cmake_prefix())));
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg("Release"));

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    if let Some(torch_lib_dir) = torch_lib_dir() {
        println!("cargo:rustc-link-search=native={torch_lib_dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{torch_lib_dir}");
    }
    println!("cargo:rustc-link-lib=static=cmz_engine");
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-lib=dylib=c10");
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={}", cpp_dir.display());
}

fn run(command: &mut Command) {
    let status = command.status().expect("failed to start build command");
    if !status.success() {
        panic!("build command failed with status {status}");
    }
}

fn torch_cmake_prefix() -> String {
    if let Ok(value) = env::var("TORCH_CMAKE_PREFIX_PATH") {
        return value;
    }
    let output = Command::new("python3")
        .arg("-c")
        .arg("import torch; print(torch.utils.cmake_prefix_path)")
        .output()
        .expect("python3 with torch is required to locate LibTorch CMake config");
    if !output.status.success() {
        panic!("failed to locate LibTorch CMake config through python3");
    }
    String::from_utf8(output.stdout)
        .expect("LibTorch CMake prefix must be UTF-8")
        .trim()
        .to_string()
}

fn torch_lib_dir() -> Option<String> {
    let output = Command::new("python3")
        .arg("-c")
        .arg("import pathlib, torch; print(pathlib.Path(torch.__file__).parent / 'lib')")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}
