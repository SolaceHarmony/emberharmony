use kcoro::{Control, Scope};

#[test]
fn parent_control_is_inherited_without_walking_children() {
    let root = Scope::root();
    let child = root.child();
    let grandchild = child.child();
    let sibling = root.child();
    let initial = grandchild.snapshot();
    assert_eq!(initial.control, Control::Running);

    assert!(root.pause());
    let paused = grandchild.snapshot();
    assert_eq!(paused.control, Control::Paused);
    assert!(paused.epoch > initial.epoch);
    assert!(root.resume());
    assert_eq!(grandchild.snapshot().control, Control::Running);

    assert!(child.cancel());
    assert_eq!(child.snapshot().control, Control::Canceled);
    assert_eq!(grandchild.snapshot().control, Control::Canceled);
    assert_eq!(sibling.snapshot().control, Control::Running);
    assert!(!child.resume());
    assert!(!child.pause());
}

#[test]
fn park_is_not_encoded_as_scope_pause() {
    let scope = Scope::root();
    let before = scope.snapshot();
    let child = scope.child();
    assert_eq!(scope.snapshot(), before);
    assert_eq!(child.snapshot().control, Control::Running);
}
