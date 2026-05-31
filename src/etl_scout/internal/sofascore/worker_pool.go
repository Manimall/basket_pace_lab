package sofascore

import (
	"context"
	"sync"

	"basket_pace_lab/etl_scout/internal/model"
)

// Result is the outcome of fetching one event. Exactly one of Stats / Err is
// non-nil. EventID is always set so callers can correlate out-of-order results.
type Result struct {
	EventID int
	Stats   *model.StatisticsResponse
	Err     error
}

// FetchManyFunc fans event IDs out across cfg.Workers goroutines and streams
// each Result to handle as it arrives, so callers can persist progress
// incrementally instead of buffering the whole batch. handle is invoked
// serially (one result at a time), so it needs no locking and "consecutive"
// counting is meaningful when Workers == 1.
//
// Returning false from handle stops the run: dispatch halts, in-flight workers
// drain, and FetchManyFunc returns. Throttling and retries live in
// FetchStatistics, so the pool size only controls parallelism.
func (c *Client) FetchManyFunc(ctx context.Context, eventIDs []int, handle func(Result) bool) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobs := make(chan int)
	results := make(chan Result)

	var wg sync.WaitGroup
	for i := 0; i < c.cfg.Workers; i++ {
		wg.Add(1)
		go c.worker(ctx, &wg, jobs, results)
	}

	// Dispatcher: feed IDs until exhausted or the context is cancelled.
	go func() {
		defer close(jobs)
		for _, id := range eventIDs {
			select {
			case <-ctx.Done():
				return
			case jobs <- id:
			}
		}
	}()

	// Closer: once every worker has returned, no more results will be sent.
	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		if !handle(r) {
			cancel()
			//nolint:revive // drain so workers blocked on send can exit
			for range results {
			}
			return
		}
	}
}

// FetchMany collects every Result into a slice (order not guaranteed). Kept for
// callers that don't need streaming; built on FetchManyFunc.
func (c *Client) FetchMany(ctx context.Context, eventIDs []int) []Result {
	collected := make([]Result, 0, len(eventIDs))
	c.FetchManyFunc(ctx, eventIDs, func(r Result) bool {
		collected = append(collected, r)
		return true
	})
	return collected
}

// worker pulls event IDs off the jobs channel and pushes one Result each.
func (c *Client) worker(
	ctx context.Context,
	wg *sync.WaitGroup,
	jobs <-chan int,
	results chan<- Result,
) {
	defer wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case id, ok := <-jobs:
			if !ok {
				return
			}
			stats, err := c.FetchStatistics(ctx, id)
			select {
			case <-ctx.Done():
				return
			case results <- Result{EventID: id, Stats: stats, Err: err}:
			}
		}
	}
}
