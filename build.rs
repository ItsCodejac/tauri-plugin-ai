fn main() {
    tauri_plugin::Builder::new(&[
        "complete",
        "stream",
        "list_models",
        "load_model",
        "unload_model",
        "set_api_key",
        "get_providers",
        "infer",
        "list_backends",
    ])
    .build();
}
