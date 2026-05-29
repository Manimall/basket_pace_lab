package ctxutil

import (
	"context"
	"testing"
	"time"
)

func TestSleepCompletes(t *testing.T) {
	start := time.Now()
	if err := Sleep(context.Background(), 20*time.Millisecond); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if elapsed := time.Since(start); elapsed < 15*time.Millisecond {
		t.Errorf("returned too early: %v", elapsed)
	}
}

func TestSleepReturnsImmediatelyOnCancel(t *testing.T) {
	// A long sleep must abort the instant the context is cancelled — this is the
	// graceful-shutdown guarantee for the -loop cooldown.
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	err := Sleep(ctx, time.Hour)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected context cancellation error")
	}
	if elapsed > 500*time.Millisecond {
		t.Errorf("did not abort promptly on cancel: waited %v", elapsed)
	}
}

func TestSleepRespectsAlreadyCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled before the call

	if err := Sleep(ctx, time.Hour); err == nil {
		t.Fatal("expected error for already-cancelled context")
	}
}
