use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::Duration;

extern "C" {
    fn kc_port_wait_u32_prepare(address: *mut u32, out: *mut *mut c_void) -> i32;
    fn kc_port_wait_u32(word: *mut c_void, expected: u32, deadline_ns: u64) -> i32;
    fn kc_port_wake_u32_all(word: *mut c_void);
    fn kc_port_wait_u32_release(word: *mut c_void);
}

#[test]
fn cargo_links_zero_spin_expected_value_waits() {
    kcoro_sys::link_anchor();
    let word = Box::into_raw(Box::new(0_u32));
    let mut wait = std::ptr::null_mut();
    assert_eq!(unsafe { kc_port_wait_u32_prepare(word, &mut wait) }, 0);
    assert!(!wait.is_null());

    // A publication before the waiter arrives is observed immediately.
    unsafe { AtomicU32::from_ptr(word) }.fetch_add(1, Ordering::Release);
    assert_eq!(unsafe { kc_port_wait_u32(wait, 0, 0) }, 0);

    let (send, recv) = mpsc::channel();
    let handle = wait as usize;
    let waiter = std::thread::spawn(move || {
        let result = unsafe { kc_port_wait_u32(handle as *mut c_void, 1, 0) };
        send.send(result).unwrap();
    });
    std::thread::sleep(Duration::from_millis(20));
    assert!(
        recv.try_recv().is_err(),
        "waiter did not park on an unchanged word"
    );
    unsafe { AtomicU32::from_ptr(word) }.fetch_add(1, Ordering::Release);
    unsafe { kc_port_wake_u32_all(wait) };
    assert_eq!(recv.recv_timeout(Duration::from_secs(2)).unwrap(), 0);
    waiter.join().unwrap();

    let (send, recv) = mpsc::channel();
    let handle = wait as usize;
    let closing = std::thread::spawn(move || {
        let result = unsafe { kc_port_wait_u32(handle as *mut c_void, 2, 0) };
        send.send(result).unwrap();
    });
    std::thread::sleep(Duration::from_millis(20));
    unsafe { kc_port_wait_u32_release(wait) };
    assert_eq!(recv.recv_timeout(Duration::from_secs(2)).unwrap(), 0);
    closing.join().unwrap();

    // SAFETY: registration is released and every waiter has joined.
    unsafe { drop(Box::from_raw(word)) };
}
