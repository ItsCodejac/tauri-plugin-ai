fn main() {
    tauri_plugin::Builder::new(&[
        "complete",
        "stream",
        "cancel_stream",
        "list_models",
        "load_model",
        "unload_model",
        "set_api_key",
        "get_api_key",
        "get_providers",
        "remove_api_key",
        "infer",
        "list_backends",
    ])
    .build();
}
