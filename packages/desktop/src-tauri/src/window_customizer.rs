use tauri::{plugin::Plugin, Manager, Runtime, Window};

pub struct PinchZoomDisablePlugin;

impl Default for PinchZoomDisablePlugin {
    fn default() -> Self {
        Self
    }
}

impl<R: Runtime> Plugin<R> for PinchZoomDisablePlugin {
    fn name(&self) -> &'static str {
        "Does not matter here"
    }

    fn window_created(&mut self, window: Window<R>) {
        let Some(webview_window) = window.get_webview_window(window.label()) else {
            return;
        };

        let _ = webview_window.with_webview(|_webview| {
            #[cfg(target_os = "linux")]
            unsafe {
                use webkit2gtk::glib::gobject_ffi;
                use webkit2gtk::glib::translate::ToGlibPtr;

                // Use raw g_object_get_data instead of ObjectExt::data::<T>() because
                // the gesture data is attached by WebKitGTK's C code, not Rust — it
                // lacks TypeId metadata so the typed wrapper would always return None.
                let obj: *mut gobject_ffi::GObject =
                    _webview.inner().to_glib_none().0 as *mut _;
                let data = gobject_ffi::g_object_get_data(
                    obj,
                    b"wk-view-zoom-gesture\0".as_ptr() as *const _,
                );
                if !data.is_null() {
                    gobject_ffi::g_signal_handlers_destroy(data as *mut gobject_ffi::GObject);
                }
            }

            #[cfg(target_os = "macos")]
            unsafe {
                use objc2::rc::Retained;
                use objc2_web_kit::WKWebView;

                // Get the WKWebView pointer and disable magnification gestures
                // This prevents Cmd+Ctrl+scroll and pinch-to-zoom from changing the zoom level
                let wk_webview: Retained<WKWebView> =
                    Retained::retain(_webview.inner().cast()).unwrap();
                wk_webview.setAllowsMagnification(false);
            }
        });
    }
}
