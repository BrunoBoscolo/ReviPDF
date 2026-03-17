// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init()) // <-- ADICIONE ESTA LINHA AQUI
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}