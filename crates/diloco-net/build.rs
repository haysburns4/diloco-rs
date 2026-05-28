fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Point prost's codegen at the protoc binary bundled by protoc-bin-vendored
    // so nothing has to be installed on the build machine.
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);

    println!("cargo:rerun-if-changed=proto/diloco.proto");
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/diloco.proto"], &["proto"])?;
    Ok(())
}
