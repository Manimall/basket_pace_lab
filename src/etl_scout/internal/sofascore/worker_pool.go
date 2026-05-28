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

// FetchMany fans event IDs out across cfg.Workers goroutines and collects the
// results. Throttling is handled inside FetchStatistics via the shared ticker,
// so the pool size controls parallelism while the rate stays globally bounded.
//
// The returned slice has one Result per input ID; order is not guaranteed.
// Cancelling ctx stops dispatch and drains in-flight work.
func (c *Client) FetchMany(ctx context.Context, eventIDs []int) []Result {
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

	collected := make([]Result, 0, len(eventIDs))
	for r := range results {
		collected = append(collected, r)
	}
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
