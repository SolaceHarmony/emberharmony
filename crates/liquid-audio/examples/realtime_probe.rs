use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Duration;

use liquid_audio::{RealtimePipeline, Utterance, VoiceEngine, VoiceEvent};

fn utt(n: usize) -> Utterance {
    Utterance {
        samples: vec![0.0; n],
        rate: 16_000,
    }
}

fn recv_event(rx: &crossbeam_channel::Receiver<VoiceEvent>) -> Result<VoiceEvent, String> {
    rx.recv_timeout(Duration::from_secs(5))
        .map_err(|e| format!("timed out waiting for event: {e}"))
}

fn expect_turn_complete(rx: &crossbeam_channel::Receiver<VoiceEvent>) -> Result<(), String> {
    match recv_event(rx)? {
        VoiceEvent::TurnComplete => Ok(()),
        ev => Err(format!("expected TurnComplete, got {ev:?}")),
    }
}

struct BlockingEngine {
    entered: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}

impl VoiceEngine for BlockingEngine {
    fn respond(
        &mut self,
        _: &Utterance,
        _: &AtomicBool,
        _: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        let _ = self.entered.send(());
        self.release
            .recv_timeout(Duration::from_secs(5))
            .map_err(|e| format!("release wait failed: {e}"))?;
        Ok(true)
    }
}

fn bounded_queue_probe() -> Result<(), String> {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let pipe = RealtimePipeline::spawn(BlockingEngine {
        entered: entered_tx,
        release: release_rx,
    })?;
    let handle = pipe
        .handle()
        .ok_or("expected live realtime pipeline handle")?;
    let clone = handle.clone();

    if !handle.submit(utt(1)) {
        return Err("first utterance should enter the worker".into());
    }
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .map_err(|e| format!("worker did not start first turn: {e}"))?;

    if !clone.submit(utt(2)) {
        return Err("one pending utterance should be accepted".into());
    }
    if handle.submit(utt(3)) {
        return Err("third utterance bypassed bounded-queue backpressure".into());
    }

    release_tx
        .send(())
        .map_err(|e| format!("failed to release first turn: {e}"))?;
    expect_turn_complete(pipe.events())?;
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .map_err(|e| format!("worker did not start pending turn: {e}"))?;
    release_tx
        .send(())
        .map_err(|e| format!("failed to release second turn: {e}"))?;
    expect_turn_complete(pipe.events())?;
    Ok(())
}

struct LoopEngine;

impl VoiceEngine for LoopEngine {
    fn respond(
        &mut self,
        _: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        for _ in 0..100_000 {
            if cancel.load(Ordering::Acquire) {
                return Ok(false);
            }
            emit(VoiceEvent::Audio { pcm: vec![0.0], rate: 24_000 });
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(true)
    }
}

fn handle_interrupt_probe() -> Result<(), String> {
    let pipe = RealtimePipeline::spawn(LoopEngine)?;
    let handle = pipe
        .handle()
        .ok_or("expected live realtime pipeline handle")?;

    if !handle.submit(utt(1)) {
        return Err("utterance should enter interrupt probe".into());
    }
    match recv_event(pipe.events())? {
        VoiceEvent::Audio { pcm: _, .. } => {}
        ev => {
            return Err(format!(
                "expected streaming audio before interrupt, got {ev:?}"
            ))
        }
    }

    handle.interrupt();
    for _ in 0..512 {
        match recv_event(pipe.events())? {
            VoiceEvent::Interrupted => return Ok(()),
            VoiceEvent::TurnComplete => {
                return Err("barge-in completed the turn instead of interrupting".into());
            }
            _ => {}
        }
    }
    Err("engine kept streaming after interrupt".into())
}

fn main() -> Result<(), String> {
    bounded_queue_probe()?;
    handle_interrupt_probe()?;
    println!("realtime pipeline probe ok");
    Ok(())
}
