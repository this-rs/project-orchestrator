use super::*;

#[test]
fn releasing_an_unstarted_turn_clears_the_claim_and_says_so() {
    let is_streaming = AtomicBool::new(true);
    let (tx, mut rx) = broadcast::channel(4);
    release_unstarted_turn(&is_streaming, &tx, &None, "s-1");
    assert!(!is_streaming.load(Ordering::SeqCst));
    match rx.try_recv() {
        Ok(ChatEvent::StreamingStatus { is_streaming }) => assert!(!is_streaming),
        other => panic!("expected StreamingStatus(false), got {other:?}"),
    }
}
