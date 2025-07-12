use std::env;
use cc::Build;
use glob::glob;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let cuda_path = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .expect("CUDA_HOME or CUDA_PATH must be set");

        let cu_files: Vec<_> = glob("src/ops/cuda/*.cu").expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect();
        
        let out_dir = env::var("OUT_DIR").unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir);

        for file in &cu_files {
            Build::new()
                .cuda(true)
                .flag("-arch=sm_70")
                .flag("-gencode=arch=compute_70,code=sm_70")
                .file(file)
                .compile(&format!("kernel_{}", file.file_stem().unwrap().to_str().unwrap()));
        }

        let lib_dir = if cfg!(target_os = "windows") { "lib/x64" } else { "lib64" };
        println!("cargo:rustc-link-search=native={}/{}", cuda_path, lib_dir);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rerun-if-changed=build.rs");
        for file in &cu_files {
            println!("cargo:rerun-if-changed={}", file.display());
        }
    }
} 