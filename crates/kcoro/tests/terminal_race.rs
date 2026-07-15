use kcoro::{promise, Resolver};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Barrier};
use std::thread;

const ITERATIONS: usize = 100_000;

fn contender(
    input: Receiver<Resolver<u8>>,
    value: u8,
    start: Arc<Barrier>,
    done: Arc<Barrier>,
    wins: Arc<AtomicUsize>,
) {
    while let Ok(resolver) = input.recv() {
        start.wait();
        if resolver.try_resolve(value).is_ok() {
            wins.fetch_add(1, Ordering::Relaxed);
        }
        done.wait();
    }
}

fn send(sender: &SyncSender<Resolver<u8>>, resolver: Resolver<u8>) {
    sender.send(resolver).unwrap();
}

#[test]
fn one_hundred_thousand_terminal_races_publish_once() {
    let start = Arc::new(Barrier::new(3));
    let done = Arc::new(Barrier::new(3));
    let left_wins = Arc::new(AtomicUsize::new(0));
    let right_wins = Arc::new(AtomicUsize::new(0));
    let (left_tx, left_rx) = mpsc::sync_channel(1);
    let (right_tx, right_rx) = mpsc::sync_channel(1);

    let left = {
        let start = start.clone();
        let done = done.clone();
        let wins = left_wins.clone();
        thread::spawn(move || contender(left_rx, 1, start, done, wins))
    };
    let right = {
        let start = start.clone();
        let done = done.clone();
        let wins = right_wins.clone();
        thread::spawn(move || contender(right_rx, 2, start, done, wins))
    };

    for _ in 0..ITERATIONS {
        let (promise, resolver) = promise();
        send(&left_tx, resolver.clone());
        send(&right_tx, resolver);
        start.wait();
        done.wait();
        assert!(matches!(promise.wait(), 1 | 2));
    }
    drop(left_tx);
    drop(right_tx);
    left.join().unwrap();
    right.join().unwrap();

    assert_eq!(
        left_wins.load(Ordering::Relaxed) + right_wins.load(Ordering::Relaxed),
        ITERATIONS
    );
    assert!(left_wins.load(Ordering::Relaxed) > 0);
    assert!(right_wins.load(Ordering::Relaxed) > 0);
}

#[test]
fn either_terminal_cause_can_own_the_claim() {
    let (first, first_resolver) = promise();
    let first_loser = first_resolver.clone();
    assert_eq!(first_resolver.try_resolve(1), Ok(()));
    assert_eq!(first_loser.try_resolve(2), Err(2));
    assert_eq!(first.wait(), 1);

    let (second, second_resolver) = promise();
    let second_winner = second_resolver.clone();
    assert_eq!(second_winner.try_resolve(2), Ok(()));
    assert_eq!(second_resolver.try_resolve(1), Err(1));
    assert_eq!(second.wait(), 2);
}
