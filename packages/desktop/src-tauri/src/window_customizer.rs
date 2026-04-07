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
                // Use glib re-exported by webkit2gtk — avoids direct dependency on
                // the unmaintained `gtk` (GTK3) crate and its pinned glib 0.18.x.
                use webkit2gtk::glib::prelude::ObjectExt;
                use webkit2gtk::glib::gobject_ffi;

                // The type parameter is irrelevant — we only need the raw pointer.
                // `data()` performs g_object_get_data under the hood; the key string
                // "wk-view-zoom-gesture" is what identifies the gesture handler.
                if let Some(data) = _webview.inner().data::<u8>("wk-view-zoom-gesture") {
                    gobject_ffi::g_signal_handlers_destroy(data.as_ptr().cast());
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
