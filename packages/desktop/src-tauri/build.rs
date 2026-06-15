fn main() {
    // libwebrtc bundles Objective-C categories on NSString that the macOS
    // linker strips without -ObjC, causing "unrecognized selector" crashes
    // at runtime (e.g. +[NSString stringForAbslStringView:]).
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-ObjC");
    }
    tauri_build::build()
}
