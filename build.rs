fn main() {
    #[cfg(feature = "cuda")]
    {
        // Trigger rebuild
        use glob::glob;
        use std::env;
        use std::fs;
        use std::path::PathBuf;

        let cuda_path = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        // Optional Windows SDK include paths (ucrt/shared/um)
        let mut windows_sdk_includes: Vec<String> = Vec::new();

        if cfg!(target_os = "windows") {
            if let Ok(entries) = fs::read_dir(r"C:\Program Files (x86)\Windows Kits\10\Include") {
                let mut candidates: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect();
                candidates.sort();
                if let Some(best) = candidates.last() {
                    let ucrt = best.join("ucrt");
                    let shared = best.join("shared");
                    let um = best.join("um");
                    let mut parts: Vec<String> = Vec::new();
                    let ucrt_s = ucrt.to_string_lossy().into_owned();
                    let shared_s = shared.to_string_lossy().into_owned();
                    let um_s = um.to_string_lossy().into_owned();
                    parts.push(ucrt_s.clone());
                    parts.push(shared_s.clone());
                    parts.push(um_s.clone());
                    windows_sdk_includes.push(ucrt_s);
                    windows_sdk_includes.push(shared_s);
                    windows_sdk_includes.push(um_s);
                    if let Ok(old) = env::var("INCLUDE") {
                        if !old.is_empty() {
                            parts.push(old);
                        }
                    }
                    let include_value = parts.join(";");
                    env::set_var("INCLUDE", include_value);
                }
            }
        }

        let cu_files: Vec<_> = glob("src/**/*.cu")
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect();

        for file in &cu_files {
            println!("cargo:rerun-if-changed={}", file.display());
        }

        let out_dir = env::var("OUT_DIR").unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);

        // Select CUDA arch from env or default to sm_70
        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_70".to_string());
        for file in &cu_files {
            let stem = file.file_stem().unwrap().to_str().unwrap();
            let lib_name = if stem == "mobius" || stem == "poincare" {
                format!("kernel_{}", stem)
            } else {
                stem.to_string()
            };

            // Use cc::Build for compiling CUDA files
            let mut build = cc::Build::new();

            build
                .cuda(true)
                .flag(&format!("-arch={}", arch))
                .include(format!("{}/include", cuda_path));

            // On Windows, force the *host* compiler to treat sources as UTF-8
            if cfg!(target_os = "windows") {
                build.flag("-Xcompiler=/utf-8");
            }

            build.file(file).compile(&lib_name);

            println!("cargo:rustc-link-lib=static={}", lib_name);
        }

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
}
