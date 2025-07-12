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

        for file in &cu_files {
            Build::new()
                .cuda(true)
                .flag("-arch=sm_70")
                .include(format!("{}/include", cuda_path))
                .file(file)
                .compile(
                    file.file_stem()
                        .unwrap()
                        .to_str()
                        .unwrap()
                );
        }

        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=kernel_mobius");
        println!("cargo:rustc-link-lib=static=kernel_poincare");
        println!("cargo:rustc-link-lib=cudart");
    }
} 