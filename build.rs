use std::env;
use glob::glob;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let cuda_path = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        let cu_files: Vec<_> = glob("src/**/*.cu")
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect();

        let out_dir = env::var("OUT_DIR").unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);

        for file in &cu_files {
            let stem = file.file_stem().unwrap().to_str().unwrap();
            let lib_name = if stem == "mobius" || stem == "poincare" {
                format!("kernel_{}", stem)
            } else {
                stem.to_string()
            };
            
            // Use cc::Build for compiling CUDA files
            cc::Build::new()
                .cuda(true)
                .flag("-arch=sm_70")
                .include(format!("{}/include", cuda_path))
                .file(file)
                .compile(&lib_name);

            println!("cargo:rustc-link-lib=static={}", lib_name);
        }

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
} 