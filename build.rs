use std::env;
use cc::Build;
use glob::glob;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let cuda_path = env::var("CUDA_PATH").expect("CUDA_PATH environment variable not set");
        
        let cu_files: Vec<_> = glob("src/ops/cuda/*.cu").expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect();

        for file in &cu_files {
        Build::new()
            .cuda(true)
                .flag("-arch=sm_70") // Adjust for your target architecture
                .file(file)
                .compile(&format!("kernel_{}", file.file_stem().unwrap().to_str().unwrap()));
        }

        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
    }
} 