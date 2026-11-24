//! Embedded static web assets for Caelum's built-in Chat UI (`/`, `/chat`) and Docs (`/docs`).
//!
//! All HTML, CSS, and JS files live cleanly separated under `caelum/web/` and are compiled
//! directly into the single Rust binary via `include_str!`, preserving the 100% zero-dependency
//! deployment guarantee while remaining editable as standalone web files.

pub const THEME_CSS: &str = include_str!("../../../web/shared/theme.css");

pub const CHAT_HTML: &str = include_str!("../../../web/chat/index.html");
pub const CHAT_CSS: &str = include_str!("../../../web/chat/chat.css");
pub const CHAT_JS: &str = include_str!("../../../web/chat/chat.js");

pub const DOCS_HTML: &str = include_str!("../../../web/docs/index.html");
pub const DOCS_CSS: &str = include_str!("../../../web/docs/docs.css");
pub const DOCS_JS: &str = include_str!("../../../web/docs/docs.js");

/// Resolves an HTTP request path to `(Content-Type, body)` for built-in web UI & assets.
pub fn resolve(path: &str) -> Option<(&'static str, &'static str)> {
    match path {
        "/" | "/chat" | "/chat/" => Some(("text/html; charset=utf-8", CHAT_HTML)),
        "/docs" | "/docs/" => Some(("text/html; charset=utf-8", DOCS_HTML)),
        "/shared/theme.css" => Some(("text/css; charset=utf-8", THEME_CSS)),
        "/chat.css" | "/chat/chat.css" => Some(("text/css; charset=utf-8", CHAT_CSS)),
        "/chat.js" | "/chat/chat.js" => Some(("application/javascript; charset=utf-8", CHAT_JS)),
        "/docs.css" | "/docs/docs.css" => Some(("text/css; charset=utf-8", DOCS_CSS)),
        "/docs.js" | "/docs/docs.js" => Some(("application/javascript; charset=utf-8", DOCS_JS)),
        _ => None,
    }
}
