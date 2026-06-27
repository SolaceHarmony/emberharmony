//! Dump every Rust fn signature (top-level, inherent-impl, trait-decl, trait-impl)
//! from a source tree as JSON, using `syn` for an exact parse. Skips `#[cfg(test)]`
//! modules. Output: a JSON array of
//!   { file, container, kind, name, has_self, args:[{pat,ty}], ret }
//! where `container` is the impl/trait type (empty for free functions).

use quote::ToTokens;
use serde_json::{json, Value};
use walkdir::WalkDir;

fn tok<T: ToTokens>(t: &T) -> String {
    // Normalize whitespace so types compare/read cleanly.
    let s = t.to_token_stream().to_string();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sig_value(sig: &syn::Signature, container: &str, kind: &str, file: &str) -> Value {
    let mut has_self = false;
    let mut args = Vec::new();
    for input in &sig.inputs {
        match input {
            syn::FnArg::Receiver(_) => has_self = true,
            syn::FnArg::Typed(pt) => {
                args.push(json!({ "pat": tok(&pt.pat), "ty": tok(&pt.ty) }));
            }
        }
    }
    let ret = match &sig.output {
        syn::ReturnType::Default => "()".to_string(),
        syn::ReturnType::Type(_, t) => tok(t),
    };
    json!({
        "file": file,
        "container": container,
        "kind": kind,
        "name": sig.ident.to_string(),
        "has_self": has_self,
        "args": args,
        "ret": ret,
    })
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        let s = a.to_token_stream().to_string();
        s.contains("cfg") && s.contains("test")
    })
}

fn walk(items: &[syn::Item], file: &str, out: &mut Vec<Value>) {
    for item in items {
        match item {
            syn::Item::Fn(f) => out.push(sig_value(&f.sig, "", "fn", file)),
            syn::Item::Impl(im) => {
                let ty = tok(&im.self_ty);
                let kind = if im.trait_.is_some() { "trait_impl_method" } else { "method" };
                for ii in &im.items {
                    if let syn::ImplItem::Fn(m) = ii {
                        out.push(sig_value(&m.sig, &ty, kind, file));
                    }
                }
            }
            syn::Item::Trait(tr) => {
                let cont = tr.ident.to_string();
                for ti in &tr.items {
                    if let syn::TraitItem::Fn(m) = ti {
                        out.push(sig_value(&m.sig, &cont, "trait_method", file));
                    }
                }
            }
            syn::Item::Mod(m) => {
                if is_cfg_test(&m.attrs) {
                    continue; // skip test modules
                }
                if let Some((_, inner)) = &m.content {
                    walk(inner, file, out);
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let root = std::env::args().nth(1).expect("usage: rust_sig_dump <src-root>");
    let mut out: Vec<Value> = Vec::new();
    for entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = match std::fs::read_to_string(p) {
            Ok(t) => t,
            Err(_) => continue,
        };
        match syn::parse_file(&text) {
            Ok(f) => {
                let rel = p.strip_prefix(&root).unwrap_or(p).to_string_lossy().to_string();
                walk(&f.items, &rel, &mut out);
            }
            Err(e) => eprintln!("parse fail {}: {e}", p.display()),
        }
    }
    println!("{}", serde_json::to_string(&out).unwrap());
}
