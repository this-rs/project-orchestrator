fn main() {
    let src_dir = std::path::Path::new("src");

    let mut c_config = cc::Build::new();
    c_config.std("c11").include(src_dir);

    #[cfg(target_env = "msvc")]
    c_config.flag("-utf-8");

    let parser_path = src_dir.join("parser.c");
    c_config.file(&parser_path);
    println!("cargo:rerun-if-changed={}", parser_path.to_str().unwrap());

    // Scanner is C++ (.cc) for this grammar
    let mut cpp_config = cc::Build::new();
    cpp_config.cpp(true).include(src_dir);

    #[cfg(target_env = "msvc")]
    cpp_config.flag("-utf-8");

    let scanner_path = src_dir.join("scanner.cc");
    cpp_config.file(&scanner_path);
    println!("cargo:rerun-if-changed={}", scanner_path.to_str().unwrap());

    c_config.compile("tree-sitter-hcl-parser");
    cpp_config.compile("tree-sitter-hcl-scanner");
}
