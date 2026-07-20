use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

extern "C" {
    fn kc_port_wait_u32_prepare(address: *mut u32, out: *mut *mut c_void) -> i32;
    fn kc_port_wait_u32(word: *mut c_void, expected: u32) -> i32;
    fn kc_port_wake_u32_all(word: *mut c_void);
    fn kc_port_wait_u32_release(word: *mut c_void);
}

#[test]
fn private_idle_address_dormancy_has_no_deadline_surface() {
    let header = include_str!("../vendor/kcoro_arena/include/kc_port.h");
    let begin = header.find("int kc_port_wait_u32_prepare").unwrap();
    let end = header[begin..]
        .find("int kc_port_thread_create")
        .map(|offset| begin + offset)
        .unwrap();
    let surface = &header[begin..end];
    assert!(!surface.contains("deadline"));
    assert!(!surface.contains("timed"));

    let port = include_str!("../vendor/kcoro_arena/port/posix.c");
    let enter = port.find("static int kc_wait_enter").unwrap();
    let closing = port[enter..]
        .find("static int kc_wait_is_closing")
        .map(|offset| enter + offset)
        .unwrap();
    let admission = &port[enter..closing];
    assert!(!admission.contains("compare_exchange"));
    assert!(!admission.contains("for (;;)") && !admission.contains("while ("));
    let begin = port.find("int kc_port_wait_u32(").unwrap();
    let end = port[begin..]
        .find("static void kc_wait_wake_native")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &port[begin..end];
    assert!(!body.contains("deadline"));
    assert!(!body.contains("timeout"));

    kcoro_sys::link_anchor();
    let word = Box::into_raw(Box::new(0_u32));
    let mut wait = std::ptr::null_mut();
    assert_eq!(unsafe { kc_port_wait_u32_prepare(word, &mut wait) }, 0);
    assert!(!wait.is_null());

    // A publication before the resident worker becomes dormant is observed
    // immediately.
    unsafe { AtomicU32::from_ptr(word) }.fetch_add(1, Ordering::Release);
    assert_eq!(unsafe { kc_port_wait_u32(wait, 0) }, 0);

    let (send, recv) = mpsc::channel();
    let handle = wait as usize;
    let sleeper = std::thread::spawn(move || {
        let result = unsafe { kc_port_wait_u32(handle as *mut c_void, 1) };
        send.send(result).unwrap();
    });
    std::thread::sleep(Duration::from_millis(20));
    assert!(
        recv.try_recv().is_err(),
        "idle worker did not become dormant on an unchanged word"
    );
    unsafe { AtomicU32::from_ptr(word) }.fetch_add(1, Ordering::Release);
    unsafe { kc_port_wake_u32_all(wait) };
    assert_eq!(recv.recv_timeout(Duration::from_secs(2)).unwrap(), 0);
    sleeper.join().unwrap();

    let (send, recv) = mpsc::channel();
    let handle = wait as usize;
    let closing = std::thread::spawn(move || {
        let result = unsafe { kc_port_wait_u32(handle as *mut c_void, 2) };
        send.send(result).unwrap();
    });
    std::thread::sleep(Duration::from_millis(20));
    unsafe { kc_port_wait_u32_release(wait) };
    assert_eq!(recv.recv_timeout(Duration::from_secs(2)).unwrap(), 0);
    closing.join().unwrap();

    // SAFETY: registration is released and every waiter has joined.
    unsafe { drop(Box::from_raw(word)) };
}

#[test]
fn release_drains_every_idle_worker_admitted_through_the_packed_gate() {
    kcoro_sys::link_anchor();
    let word = Box::into_raw(Box::new(0_u32));
    let mut wait = std::ptr::null_mut();
    assert_eq!(unsafe { kc_port_wait_u32_prepare(word, &mut wait) }, 0);

    const WORKERS: usize = 32;
    let start = Arc::new(Barrier::new(WORKERS + 1));
    let (send, recv) = mpsc::channel();
    let handles = (0..WORKERS)
        .map(|_| {
            let start = Arc::clone(&start);
            let send = send.clone();
            let handle = wait as usize;
            std::thread::spawn(move || {
                start.wait();
                let result = unsafe { kc_port_wait_u32(handle as *mut c_void, 0) };
                send.send(result).unwrap();
            })
        })
        .collect::<Vec<_>>();
    drop(send);
    start.wait();
    std::thread::sleep(Duration::from_millis(20));
    unsafe { kc_port_wait_u32_release(wait) };
    for _ in 0..WORKERS {
        assert_eq!(recv.recv_timeout(Duration::from_secs(2)).unwrap(), 0);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    unsafe { drop(Box::from_raw(word)) };
}
