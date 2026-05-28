package sofascore

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"basket_pace_lab/etl_scout/internal/config"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

const fixtureStats = `{"statistics":[{"period":"ALL","groups":[]}]}`

func newTestClient(t *testing.T, baseURL string, workers, rps int) *Client {
	t.Helper()
	c, err := NewClient(config.SofascoreConfig{
		BaseURL:        baseURL,
		Workers:        workers,
		RequestsPerSec: rps,
		HTTPTimeout:    5 * time.Second,
	}, quietLogger())
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return c
}

func TestFetchManyReturnsAllResults(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&hits, 1)
		if _, err := w.Write([]byte(fixtureStats)); err != nil {
			t.Errorf("write fixture: %v", err)
		}
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, 3, 100)
	defer client.Close()

	ids := []int{1, 2, 3, 4, 5}
	results := client.FetchMany(context.Background(), ids)

	if len(results) != len(ids) {
		t.Fatalf("expected %d results, got %d", len(ids), len(results))
	}
	for _, r := range results {
		if r.Err != nil {
			t.Errorf("event %d: unexpected error %v", r.EventID, r.Err)
			continue
		}
		if len(r.Stats.Statistics) != 1 {
			t.Errorf("event %d: expected 1 period, got %d", r.EventID, len(r.Stats.Statistics))
		}
	}
	if got := atomic.LoadInt32(&hits); got != int32(len(ids)) {
		t.Errorf("expected %d HTTP hits, got %d", len(ids), got)
	}
}

func TestFetchStatisticsPropagatesHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, 1, 100)
	defer client.Close()

	_, err := client.FetchStatistics(context.Background(), 42)
	if err == nil {
		t.Fatal("expected error on HTTP 500")
	}
}

func TestFetchManyRespectsRateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(fixtureStats)); err != nil {
			t.Errorf("write fixture: %v", err)
		}
	}))
	defer srv.Close()

	// rps=10 → 100ms between ticks. 5 requests across 2 workers must still be
	// globally throttled, so total time is bounded below by the ticker.
	client := newTestClient(t, srv.URL, 2, 10)
	defer client.Close()

	start := time.Now()
	results := client.FetchMany(context.Background(), []int{1, 2, 3, 4, 5})
	elapsed := time.Since(start)

	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}
	const minExpected = 300 * time.Millisecond
	if elapsed < minExpected {
		t.Errorf("rate limit not enforced: 5 requests took %v, want >= %v", elapsed, minExpected)
	}
}

func TestFetchManyHonoursContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(fixtureStats)); err != nil {
			t.Errorf("write fixture: %v", err)
		}
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, 2, 1) // slow: 1 rps
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	results := client.FetchMany(ctx, []int{1, 2, 3, 4, 5, 6, 7, 8})
	// With a 50ms deadline and 1 rps, the run must stop well short of 8 results.
	if len(results) >= 8 {
		t.Errorf("expected cancellation to cut the run short, got %d results", len(results))
	}
}
