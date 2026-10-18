//! Native macOS window styling (rounded corners, transparent titlebar, traffic lights).
//!
//! Uses the typed `objc2` bindings (objc2-app-kit, objc2-quartz-core, objc2-web-kit), the
//! successors of the deprecated `cocoa` crate. Only public AppKit APIs: App Store compatible.
//! The only `unsafe` left is turning Tauri's raw `NSWindow` / `WKWebView` pointers into typed
//! references ([`ns_window`], [`wk_webview`]) and one WebKit setter the bindings mark unsafe.

use tauri::{AppHandle, Runtime, WebviewWindow};

#[cfg(target_os = "macos")]
use objc2::{rc::Retained, MainThreadMarker};
#[cfg(target_os = "macos")]
use objc2_app_kit::{NSWindow, NSWindowButton, NSWindowStyleMask, NSWindowTitleVisibility};
#[cfg(target_os = "macos")]
use objc2_foundation::{NSPoint, NSRect};
#[cfg(target_os = "macos")]
use objc2_web_kit::WKWebView;

/// Configuration for Traffic Lights positioning
pub struct TrafficLightsConfig {
    /// Offset in pixels from default position (positive = right, negative = left)
    pub offset_x: f64,
    /// Offset in pixels from default position (positive = down, negative = up)
    pub offset_y: f64,
}

impl Default for TrafficLightsConfig {
    fn default() -> Self {
        Self {
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}

/// The `NSWindow` hosting a Tauri webview, as a typed, retained reference.
///
/// `None` when called off the main thread (AppKit objects are main-thread-only) or when Tauri
/// hands a null pointer.
#[cfg(target_os = "macos")]
pub(crate) fn ns_window(webview: &tauri::webview::PlatformWebview) -> Option<Retained<NSWindow>> {
    if MainThreadMarker::new().is_none() {
        tracing::warn!("macOS window styling skipped: not on the main thread");
        return None;
    }
    // SAFETY: on macOS `PlatformWebview::ns_window()` is the `NSWindow*` that owns the webview,
    // alive for the whole `with_webview` callback (Tauri runs that callback on the main thread,
    // checked just above). `Retained::retain` takes its own +1 reference and accepts null.
    unsafe { Retained::retain(webview.ns_window().cast::<NSWindow>()) }
}

/// The `WKWebView` of a Tauri webview, as a typed, retained reference. Same conditions as
/// [`ns_window`].
#[cfg(target_os = "macos")]
pub(crate) fn wk_webview(webview: &tauri::webview::PlatformWebview) -> Option<Retained<WKWebView>> {
    MainThreadMarker::new()?;
    // SAFETY: on macOS `PlatformWebview::inner()` is wry's `WKWebView*`, alive for the whole
    // `with_webview` callback, on the main thread (checked above). `retain` accepts null.
    unsafe { Retained::retain(webview.inner().cast::<WKWebView>()) }
}

/// Full-size content view, titled, closable, miniaturizable, resizable: what the rounded,
/// titlebar-less look needs, on top of the window's current style.
#[cfg(target_os = "macos")]
pub(crate) fn add_modern_style_mask(ns_window: &NSWindow) {
    let style_mask = ns_window.styleMask()
        | NSWindowStyleMask::FullSizeContentView
        | NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable;
    ns_window.setStyleMask(style_mask);
}

/// Layer-backs the content view and rounds its corners (`radius` in points). `None` leaves
/// the radius unchanged (layer-backing only).
#[cfg(target_os = "macos")]
pub(crate) fn round_content_view(ns_window: &NSWindow, radius: Option<f64>) {
    let Some(content_view) = ns_window.contentView() else {
        return;
    };
    content_view.setWantsLayer(true);
    if let Some(radius) = radius {
        if let Some(layer) = content_view.layer() {
            layer.setCornerRadius(radius);
            layer.setMasksToBounds(true);
        }
    }
}

/// Sets the WKWebView's under-page background (shown during overscroll, navigation and
/// before the first paint), to avoid white flashes.
#[cfg(target_os = "macos")]
pub(crate) fn set_under_page_background(
    webview: &tauri::webview::PlatformWebview,
    color: &objc2_app_kit::NSColor,
) {
    if let Some(wk_webview) = wk_webview(webview) {
        // SAFETY: the bindings mark every WKWebView setter unsafe by default; this one takes an
        // optional NSColor, which it copies. `color` is a valid NSColor and we are on the main
        // thread (`wk_webview` returned Some).
        unsafe { wk_webview.setUnderPageBackgroundColor(Some(color)) };
    }
}

/// Enables rounded corners for the window (macOS only)
/// Uses only public APIs - App Store compatible
#[tauri::command]
pub fn enable_rounded_corners<R: Runtime>(
    _app: AppHandle<R>,
    window: WebviewWindow<R>,
    offset_x: Option<f64>,
    offset_y: Option<f64>,
) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let config = TrafficLightsConfig {
            offset_x: offset_x.unwrap_or(0.0),
            offset_y: offset_y.unwrap_or(0.0),
        };

        window
            .with_webview(move |webview| {
                let Some(ns_window) = ns_window(&webview) else {
                    return;
                };
                add_modern_style_mask(&ns_window);
                ns_window.setTitlebarAppearsTransparent(true);
                round_content_view(&ns_window, None);
                position_traffic_lights(&ns_window, config.offset_x, config.offset_y);
            })
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (window, offset_x, offset_y);
        Ok(())
    }
}

/// Enables modern window style with rounded corners and shadow
#[tauri::command]
pub fn enable_modern_window_style<R: Runtime>(
    _app: AppHandle<R>,
    window: WebviewWindow<R>,
    corner_radius: Option<f64>,
    offset_x: Option<f64>,
    offset_y: Option<f64>,
) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let config = TrafficLightsConfig {
            offset_x: offset_x.unwrap_or(0.0),
            offset_y: offset_y.unwrap_or(0.0),
        };
        let radius = corner_radius.unwrap_or(12.0);

        window
            .with_webview(move |webview| {
                let Some(ns_window) = ns_window(&webview) else {
                    return;
                };
                add_modern_style_mask(&ns_window);
                ns_window.setTitlebarAppearsTransparent(true);
                ns_window.setTitleVisibility(NSWindowTitleVisibility::Hidden);
                ns_window.setHasShadow(true);
                ns_window.setOpaque(false);
                round_content_view(&ns_window, Some(radius));
                position_traffic_lights(&ns_window, config.offset_x, config.offset_y);
            })
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (window, corner_radius, offset_x, offset_y);
        Ok(())
    }
}

/// Repositions Traffic Lights only (useful after fullscreen toggle)
#[tauri::command]
pub fn reposition_traffic_lights<R: Runtime>(
    _app: AppHandle<R>,
    window: WebviewWindow<R>,
    offset_x: Option<f64>,
    offset_y: Option<f64>,
) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let config = TrafficLightsConfig {
            offset_x: offset_x.unwrap_or(0.0),
            offset_y: offset_y.unwrap_or(0.0),
        };

        window
            .with_webview(move |webview| {
                if let Some(ns_window) = ns_window(&webview) {
                    position_traffic_lights(&ns_window, config.offset_x, config.offset_y);
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (window, offset_x, offset_y);
        Ok(())
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
/// x of the close, miniaturize and zoom buttons: 20 pt apart, starting at `20 + offset_x`.
fn traffic_light_xs(offset_x: f64) -> [f64; 3] {
    let x = 20.0 + offset_x;
    [x, x + 20.0, x + 40.0]
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
/// y of the traffic lights: positive `offset_y` moves them down (AppKit's y axis points up).
fn traffic_light_y(offset_y: f64) -> f64 {
    -offset_y
}

#[cfg(target_os = "macos")]
fn position_traffic_lights(ns_window: &NSWindow, offset_x: f64, offset_y: f64) {
    let new_y = traffic_light_y(offset_y);
    let buttons = [
        NSWindowButton::CloseButton,
        NSWindowButton::MiniaturizeButton,
        NSWindowButton::ZoomButton,
    ];
    for (kind, new_x) in buttons.into_iter().zip(traffic_light_xs(offset_x)) {
        if let Some(button) = ns_window.standardWindowButton(kind) {
            let frame = button.frame();
            button.setFrame(NSRect::new(NSPoint::new(new_x, new_y), frame.size));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traffic_lights_keep_the_historical_layout() {
        // Same numbers as the cocoa version: x = 20 + offset, then +20 and +40; y = -offset.
        assert_eq!(traffic_light_xs(0.0), [20.0, 40.0, 60.0]);
        assert_eq!(traffic_light_xs(5.0), [25.0, 45.0, 65.0]);
        assert_eq!(traffic_light_y(0.0), 0.0);
        assert_eq!(traffic_light_y(4.0), -4.0);
    }
}
