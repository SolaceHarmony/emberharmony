use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const MODE_MASK: u64 = 0b11;
const RUNNING: u64 = 0;
const PAUSED: u64 = 1;
const CANCELED: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Control {
    Running,
    Paused,
    Canceled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControlSnapshot {
    pub control: Control,
    pub epoch: u64,
}

struct Domain {
    sequence: AtomicU64,
}

struct Node {
    domain: Arc<Domain>,
    parent: Option<Arc<Node>>,
    word: AtomicU64,
}

#[derive(Clone)]
pub struct Scope {
    node: Arc<Node>,
}

impl Scope {
    pub fn root() -> Self {
        let domain = Arc::new(Domain {
            sequence: AtomicU64::new(1),
        });
        Self {
            node: Arc::new(Node {
                domain,
                parent: None,
                word: AtomicU64::new(pack(1, RUNNING)),
            }),
        }
    }

    pub fn child(&self) -> Self {
        Self {
            node: Arc::new(Node {
                domain: self.node.domain.clone(),
                parent: Some(self.node.clone()),
                word: AtomicU64::new(pack(0, RUNNING)),
            }),
        }
    }

    pub fn snapshot(&self) -> ControlSnapshot {
        let mut node = Some(self.node.as_ref());
        let mut epoch = 0;
        let mut control = Control::Running;
        while let Some(current) = node {
            let word = current.word.load(Ordering::Acquire);
            epoch = epoch.max(word >> 2);
            match word & MODE_MASK {
                CANCELED => control = Control::Canceled,
                PAUSED if control != Control::Canceled => control = Control::Paused,
                _ => {}
            }
            node = current.parent.as_deref();
        }
        ControlSnapshot { control, epoch }
    }

    pub fn pause(&self) -> bool {
        self.transition(PAUSED, false)
    }

    pub fn resume(&self) -> bool {
        self.transition(RUNNING, false)
    }

    pub fn cancel(&self) -> bool {
        self.transition(CANCELED, true)
    }

    fn transition(&self, mode: u64, terminal: bool) -> bool {
        let mut word = self.node.word.load(Ordering::Acquire);
        loop {
            let current = word & MODE_MASK;
            if current == CANCELED || current == mode {
                return false;
            }
            if terminal && mode != CANCELED {
                return false;
            }
            let epoch = self.node.domain.sequence.fetch_add(1, Ordering::AcqRel) + 1;
            match self.node.word.compare_exchange(
                word,
                pack(epoch, mode),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => word = actual,
            }
        }
    }
}

const fn pack(epoch: u64, mode: u64) -> u64 {
    (epoch << 2) | mode
}
