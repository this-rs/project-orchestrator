//! Helpers shared by the XML-in-a-ZIP extractors (`.docx`, `.pptx`).

/// Resolve one entity reference to the text it stands for.
///
/// Delegates to quick-xml rather than matching the five predefined names by
/// hand, because Office also emits numeric references (`&#8217;` for a curly
/// apostrophe) and those have to resolve the same way.
pub(super) fn resolve_entity(reference: &quick_xml::events::BytesRef<'_>) -> Option<String> {
    let name = reference.decode().ok()?;
    quick_xml::escape::unescape(&format!("&{name};"))
        .ok()
        .map(|s| s.into_owned())
}
