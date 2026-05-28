// Package sofascore is a concurrent, rate-limited client for Sofascore's
// internal box-score API.
package sofascore

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"basket_pace_lab/etl_scout/internal/config"
	"basket_pace_lab/etl_scout/internal/model"
)

// Browser-imitating headers reduce the odds of a 403 from the edge WAF.
var defaultHeaders = map[string]string{
	"Accept":          "application/json, text/plain, */*",
	"Accept-Language": "en-US,en;q=0.9",
	"Origin":          "https://www.sofascore.com",
	"Referer":         "https://www.sofascore.com/",
	"Cache-Control":   "no-cache",
}

const statusOK = http.StatusOK

// Client issues throttled GETs against the Sofascore API. A single shared
// time.Ticker enforces a global request rate regardless of worker count, so
// concurrency never translates into a request burst that trips IP blocking.
type Client struct {
	cfg     config.SofascoreConfig
	http    *http.Client
	limiter *time.Ticker
	log     *slog.Logger
}

// NewClient builds a Client. Close() must be called to release the ticker.
func NewClient(cfg config.SofascoreConfig, log *slog.Logger) *Client {
	interval := time.Second / time.Duration(cfg.RequestsPerSec)
	return &Client{
		cfg:     cfg,
		http:    &http.Client{Timeout: cfg.HTTPTimeout},
		limiter: time.NewTicker(interval),
		log:     log,
	}
}

// Close stops the rate-limiting ticker.
func (c *Client) Close() {
	c.limiter.Stop()
}

// FetchStatistics retrieves and decodes the period-split statistics for one
// event. It blocks on the shared rate limiter before issuing the request, so
// it is safe to call from many goroutines concurrently.
func (c *Client) FetchStatistics(ctx context.Context, eventID int) (*model.StatisticsResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.limiter.C:
	}

	url := fmt.Sprintf("%s/event/%d/statistics", c.cfg.BaseURL, eventID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request for event %d: %w", eventID, err)
	}
	for k, v := range defaultHeaders {
		req.Header.Set(k, v)
	}

	c.log.Debug("GET statistics", "event_id", eventID, "url", url)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request event %d: %w", eventID, err)
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			c.log.Warn("close response body", "event_id", eventID, "error", cerr)
		}
	}()

	if resp.StatusCode != statusOK {
		return nil, fmt.Errorf("event %d: unexpected status %d", eventID, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body for event %d: %w", eventID, err)
	}

	var out model.StatisticsResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("decode statistics for event %d: %w", eventID, err)
	}
	return &out, nil
}
