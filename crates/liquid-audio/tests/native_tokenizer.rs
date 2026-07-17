use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer as OracleTokenizer};

const ABI: u32 = 1;

#[repr(C)]
struct NativeTokenizer {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Default)]
struct Special {
    size: u32,
    abi_version: u32,
    im_start: u32,
    im_end: u32,
    text_end: u32,
    audio_start: u32,
    reserved: [u32; 4],
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct WorkspaceInfo {
    size: u32,
    abi_version: u32,
    max_input_bytes: u64,
    storage_bytes: u64,
    encode_calls: u64,
    reserved: [u64; 4],
}

unsafe extern "C" {
    fn lfm_tokenizer_open(
        path: *const c_char,
        out: *mut *mut NativeTokenizer,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_tokenizer_close(tokenizer: *mut NativeTokenizer);
    fn lfm_tokenizer_special(tokenizer: *const NativeTokenizer, out: *mut Special) -> i32;
    fn lfm_tokenizer_workspace_create(
        max_input_bytes: usize,
        out: *mut *mut NativeWorkspace,
    ) -> i32;
    fn lfm_tokenizer_workspace_destroy(workspace: *mut NativeWorkspace);
    fn lfm_tokenizer_workspace_info(
        workspace: *const NativeWorkspace,
        out: *mut WorkspaceInfo,
    ) -> i32;
    fn lfm_tokenizer_encode_bounded(
        tokenizer: *const NativeTokenizer,
        workspace: *mut NativeWorkspace,
        text: *const c_char,
        text_bytes: usize,
        out: *mut u32,
        out_capacity: usize,
        out_count: *mut usize,
    ) -> i32;
    fn lfm_tokenizer_encode(
        tokenizer: *const NativeTokenizer,
        text: *const c_char,
        text_bytes: usize,
        out: *mut u32,
        out_capacity: usize,
        out_count: *mut usize,
    ) -> i32;
    fn lfm_tokenizer_decode_piece(
        tokenizer: *const NativeTokenizer,
        token: u32,
        skip_special: u32,
        out: *mut u8,
        out_capacity: usize,
        out_bytes: *mut usize,
    ) -> i32;
}

struct Tokenizer(*mut NativeTokenizer);

impl Drop for Tokenizer {
    fn drop(&mut self) {
        unsafe { lfm_tokenizer_close(self.0) };
    }
}

struct Workspace(*mut NativeWorkspace);

impl Drop for Workspace {
    fn drop(&mut self) {
        unsafe { lfm_tokenizer_workspace_destroy(self.0) };
    }
}

struct Fixture {
    path: PathBuf,
    oracle: OracleTokenizer,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn fixture() -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = HashMap::new();
    for (id, value) in alphabet.into_iter().enumerate() {
        vocab.insert(value.to_string(), id as u32);
    }
    let merges = vec![
        ("t".to_string(), "e".to_string()),
        ("te".to_string(), "s".to_string()),
        ("tes".to_string(), "t".to_string()),
        ("i".to_string(), "n".to_string()),
        ("in".to_string(), "g".to_string()),
    ];
    for (left, right) in &merges {
        const MAX_VOCAB: usize = u32::MAX as usize;
        assert!(vocab.len() < MAX_VOCAB);
        let id = vocab.len() as u32;
        vocab.entry(format!("{left}{right}")).or_insert(id);
    }
    const SPECIALS: [&str; 4] = [
        "<|im_start|>",
        "<|im_end|>",
        "<|text_end|>",
        "<|audio_start|>",
    ];
    for token in SPECIALS {
        let id = vocab.len() as u32;
        vocab.insert(token.to_string(), id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .cache_capacity(0)
        .build()
        .unwrap();
    let bytelevel = ByteLevel::new(false, true, true);
    let mut oracle = OracleTokenizer::new(model);
    oracle.with_pre_tokenizer(Some(bytelevel));
    oracle.with_decoder(Some(bytelevel));
    oracle.add_special_tokens(
        &SPECIALS
            .into_iter()
            .map(|token| AddedToken::from(token, true))
            .collect::<Vec<_>>(),
    );
    let path = std::env::temp_dir().join(format!(
        "liquid-audio-native-tokenizer-{}-{}.json",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    oracle.save(&path, false).unwrap();
    Fixture { path, oracle }
}

fn open(path: &std::path::Path) -> Tokenizer {
    let path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let mut tokenizer = std::ptr::null_mut();
    let mut error = [0i8; 512];
    let status = unsafe {
        lfm_tokenizer_open(
            path.as_ptr(),
            &mut tokenizer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    assert_eq!(
        status,
        0,
        "{}",
        unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );
    assert!(!tokenizer.is_null());
    Tokenizer(tokenizer)
}

fn workspace(capacity: usize) -> Workspace {
    let mut workspace = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_tokenizer_workspace_create(capacity, &mut workspace) },
        0
    );
    assert!(!workspace.is_null());
    Workspace(workspace)
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let mut count = 0usize;
    assert_eq!(
        unsafe {
            lfm_tokenizer_encode(
                tokenizer.0,
                text.as_ptr().cast(),
                text.len(),
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        },
        if text.is_empty() { 0 } else { -28 }
    );
    let mut ids = vec![0u32; count];
    assert_eq!(
        unsafe {
            lfm_tokenizer_encode(
                tokenizer.0,
                text.as_ptr().cast(),
                text.len(),
                ids.as_mut_ptr(),
                ids.len(),
                &mut count,
            )
        },
        0
    );
    ids
}

fn encode_bounded(tokenizer: &Tokenizer, workspace: &Workspace, text: &str) -> Vec<u32> {
    let mut ids = vec![u32::MAX; text.len()];
    let mut count = 0usize;
    assert_eq!(
        unsafe {
            lfm_tokenizer_encode_bounded(
                tokenizer.0,
                workspace.0,
                text.as_ptr().cast(),
                text.len(),
                ids.as_mut_ptr(),
                ids.len(),
                &mut count,
            )
        },
        0
    );
    ids.truncate(count);
    ids
}

fn workspace_info(workspace: &Workspace) -> WorkspaceInfo {
    let mut info = WorkspaceInfo {
        size: std::mem::size_of::<WorkspaceInfo>() as u32,
        abi_version: ABI,
        ..Default::default()
    };
    assert_eq!(
        unsafe { lfm_tokenizer_workspace_info(workspace.0, &mut info) },
        0
    );
    info
}

#[test]
fn native_and_bounded_byte_bpe_match_serialized_oracle_fixture() {
    use liquid_audio as _;
    let fixture = fixture();
    let native = open(&fixture.path);
    let workspace = workspace(2048);
    let corpus = [
        "<|im_start|>system\nYou are a brief voice assistant.<|im_end|>\n",
        "<|im_start|>user\n<|audio_start|><|im_end|>\n<|im_start|>assistant\n",
        "We're testing punctuation—three digits 1234, tabs\tand\nnewlines.  ",
        "Unicode café, naïve, 東京, and emoji 🎧.",
    ];
    for text in corpus {
        let expected = fixture.oracle.encode(text, false).unwrap();
        assert_eq!(encode(&native, text), expected.get_ids(), "input: {text:?}");
        assert_eq!(
            encode_bounded(&native, &workspace, text),
            expected.get_ids(),
            "bounded input: {text:?}"
        );
    }

    let mut special = Special {
        size: std::mem::size_of::<Special>() as u32,
        abi_version: ABI,
        ..Default::default()
    };
    assert_eq!(unsafe { lfm_tokenizer_special(native.0, &mut special) }, 0);
    assert_eq!(
        special.im_start,
        fixture.oracle.token_to_id("<|im_start|>").unwrap()
    );
    assert_eq!(
        special.im_end,
        fixture.oracle.token_to_id("<|im_end|>").unwrap()
    );
    assert_eq!(
        special.audio_start,
        fixture.oracle.token_to_id("<|audio_start|>").unwrap()
    );
    assert_eq!(
        special.text_end,
        fixture.oracle.token_to_id("<|text_end|>").unwrap()
    );

    let text = "A byte-level café 🎧 round trip.";
    let ids = encode(&native, text);
    let mut decoded = Vec::new();
    for id in ids {
        let mut count = 0usize;
        assert_eq!(
            unsafe {
                lfm_tokenizer_decode_piece(native.0, id, 1, std::ptr::null_mut(), 0, &mut count)
            },
            if count == 0 { 0 } else { -28 }
        );
        let offset = decoded.len();
        decoded.resize(offset + count, 0);
        assert_eq!(
            unsafe {
                lfm_tokenizer_decode_piece(
                    native.0,
                    id,
                    1,
                    decoded[offset..].as_mut_ptr(),
                    count,
                    &mut count,
                )
            },
            0
        );
    }
    assert_eq!(decoded, text.as_bytes());
}

#[test]
fn bounded_workspace_is_fixed_and_rejects_invalid_or_oversized_input() {
    use liquid_audio as _;
    const CAPACITY: usize = 2048;
    let fixture = fixture();
    let native = open(&fixture.path);
    let workspace = workspace(CAPACITY);
    const TEXT: &str = "<|im_start|>user\ntesting testing café 🎧<|im_end|>";
    let expected = fixture.oracle.encode(TEXT, false).unwrap();
    let before = workspace_info(&workspace);
    assert_eq!(before.max_input_bytes, CAPACITY as u64);
    assert!(before.storage_bytes >= CAPACITY as u64 * std::mem::size_of::<u32>() as u64);

    for _ in 0..10_000 {
        assert_eq!(
            encode_bounded(&native, &workspace, TEXT),
            expected.get_ids()
        );
    }
    let after = workspace_info(&workspace);
    assert_eq!(after.max_input_bytes, before.max_input_bytes);
    assert_eq!(after.storage_bytes, before.storage_bytes);
    assert_eq!(after.encode_calls - before.encode_calls, 10_000);

    let invalid = [0xc3u8, 0x28];
    let mut count = 99usize;
    let mut output = [u32::MAX; CAPACITY];
    assert_eq!(
        unsafe {
            lfm_tokenizer_encode_bounded(
                native.0,
                workspace.0,
                invalid.as_ptr().cast(),
                invalid.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut count,
            )
        },
        -libc::EINVAL
    );
    assert_eq!(count, 0);

    let oversized = vec![b'x'; CAPACITY + 1];
    assert_eq!(
        unsafe {
            lfm_tokenizer_encode_bounded(
                native.0,
                workspace.0,
                oversized.as_ptr().cast(),
                oversized.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut count,
            )
        },
        -libc::ENOBUFS
    );
    assert_eq!(count, 0);

    let mut tiny = [u32::MAX; 1];
    assert_eq!(
        unsafe {
            lfm_tokenizer_encode_bounded(
                native.0,
                workspace.0,
                TEXT.as_ptr().cast(),
                TEXT.len(),
                tiny.as_mut_ptr(),
                tiny.len(),
                &mut count,
            )
        },
        -libc::ENOBUFS
    );
    assert!(count > tiny.len());
    assert_eq!(tiny, [u32::MAX]);
}
