//! Shared managed-thread primitive for the native voice layer.
//!
//! Provider sessions, LiveKit helper loops, delegated turns, and model downloads
//! all use this rather than each subsystem carrying its own orphan-prone handle
//! list.

use std::{
    sync::{Arc, Mutex},
    thread::{Builder as ThreadBuilder, JoinHandle},
};

#[derive(Clone, Default)]
pub(crate) struct ThreadManager {
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl ThreadManager {
    pub(crate) fn spawn(
        &self,
        name: &str,
        f: impl FnOnce() + Send + 'static,
    ) -> Result<(), String> {
        let handle = ThreadBuilder::new()
            .name(name.into())
            .spawn(f)
            .map_err(|e| format!("voice thread spawn failed: {e}"))?;
        self.handles
            .lock()
            .map_err(|_| "voice thread manager lock poisoned".to_string())?
            .push(handle);
        Ok(())
    }

    pub(crate) fn spawn_if_idle(
        &self,
        name: &str,
        busy: &str,
        f: impl FnOnce() + Send + 'static,
    ) -> Result<(), String> {
        self.reap()?;
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| "voice thread manager lock poisoned".to_string())?;
        if !handles.is_empty() {
            return Err(busy.into());
        }
        let handle = ThreadBuilder::new()
            .name(name.into())
            .spawn(f)
            .map_err(|e| format!("voice thread spawn failed: {e}"))?;
        handles.push(handle);
        Ok(())
    }

    pub(crate) fn reap(&self) -> Result<(), String> {
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| "voice thread manager lock poisoned".to_string())?;
        let mut live = Vec::new();
        for handle in handles.drain(..) {
            if handle.is_finished() {
                let _ = handle.join();
                continue;
            }
            live.push(handle);
        }
        *handles = live;
        Ok(())
    }

    pub(crate) fn wait(&self) -> Result<(), String> {
        let current = std::thread::current().id();
        let handles = {
            let mut handles = self
                .handles
                .lock()
                .map_err(|_| "voice thread manager lock poisoned".to_string())?;
            handles.drain(..).collect::<Vec<_>>()
        };
        let mut live = Vec::new();
        for handle in handles {
            if handle.thread().id() == current {
                live.push(handle);
                continue;
            }
            let _ = handle.join();
        }
        if !live.is_empty() {
            self.handles
                .lock()
                .map_err(|_| "voice thread manager lock poisoned".to_string())?
                .extend(live);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn tracked_len(&self) -> usize {
        self.handles.lock().unwrap().len()
    }
}

impl Drop for ThreadManager {
    fn drop(&mut self) {
        let Some(handles) = Arc::get_mut(&mut self.handles) else {
            return;
        };
        let Ok(handles) = handles.get_mut() else {
            return;
        };
        let current = std::thread::current().id();
        let mut live = Vec::new();
        for handle in handles.drain(..) {
            if handle.thread().id() == current {
                live.push(handle);
                continue;
            }
            let _ = handle.join();
        }
        handles.extend(live);
    }
}
