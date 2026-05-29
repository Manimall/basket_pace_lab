// Package ctxutil holds small context-aware helpers shared across the service.
package ctxutil

import (
	"context"
	"time"
)

// Sleep blocks for d, returning early with ctx.Err() if the context is
// cancelled first. The timer is always released, so a cancelled long sleep
// never leaks a pending timer.
func Sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
