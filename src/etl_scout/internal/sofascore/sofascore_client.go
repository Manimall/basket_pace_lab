// Package sofascore is a concurrent, rate-limited client for Sofascore's
// internal box-score API.
package sofascore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"os"
	"strings"
	"time"

	http "github.com/bogdanfinn/fhttp"
	tls_client "github.com/bogdanfinn/tls-client"
	"github.com/bogdanfinn/tls-client/profiles"

	"basket_pace_lab/etl_scout/internal/config"
	"basket_pace_lab/etl_scout/internal/model"
)

// clientProfile is the TLS/HTTP2 fingerprint we impersonate. Sofascore's Varnish
// edge fingerprints precisely: only the Chrome 124 profile over HTTP/2 passes —
// Chrome 120/131/133 and forced HTTP/1.1 all get a 403. This mirrors the Python
// side's curl_cffi impersonate="chrome124". Re-verify if Sofascore tightens.
var clientProfile = profiles.Chrome_124

// Browser-imitating headers reduce the odds of a 403 from the edge WAF. The
// User-Agent must match clientProfile so the JA3 and the UA tell one story.
var defaultHeaders = map[string]string{
	"Accept":          "application/json, text/plain, */*",
	"Accept-Language": "en-US,en;q=0.9",
	"Origin":          "https://www.sofascore.com",
	"Referer":         "https://www.sofascore.com/",
	"Cache-Control":   "no-cache",
	"User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}

const statusOK = http.StatusOK

// ErrNoStatistics signals Sofascore has no box score for the event (HTTP 404).
// Callers treat it as a soft skip, not a hard failure.
var ErrNoStatistics = errors.New("no statistics available")

// cookie is the browser-export shape we read from cookies.json.
type cookie struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// Client issues throttled GETs against the Sofascore API. A single shared
// time.Ticker enforces a global request rate regardless of worker count, so
// concurrency never translates into a request burst that trips IP blocking.
type Client struct {
	cfg          config.SofascoreConfig
	http         tls_client.HttpClient
	limiter      *time.Ticker
	log          *slog.Logger
	cookieHeader string
}

// NewClient builds a Client with a Chrome-impersonating TLS transport. Close()
// must be called to release the ticker.
func NewClient(cfg config.SofascoreConfig, log *slog.Logger) (*Client, error) {
	interval := time.Second / time.Duration(cfg.RequestsPerSec)

	timeoutSecs := int(cfg.HTTPTimeout.Seconds())
	if timeoutSecs < 1 {
		timeoutSecs = 1
	}
	httpClient, err := tls_client.NewHttpClient(tls_client.NewNoopLogger(),
		tls_client.WithClientProfile(clientProfile),
		tls_client.WithTimeoutSeconds(timeoutSecs),
		tls_client.WithNotFollowRedirects(),
	)
	if err != nil {
		return nil, fmt.Errorf("build tls client: %w", err)
	}

	cookieHeader, err := loadCookieHeader(cfg.CookiesPath)
	switch {
	case err != nil:
		log.Warn("cookie load failed; proceeding without", "path", cfg.CookiesPath, "error", err)
	case cookieHeader == "":
		log.Warn("no cookies loaded; Sofascore may return 403", "path", cfg.CookiesPath)
	default:
		log.Info("loaded sofascore cookies", "path", cfg.CookiesPath)
	}

	return &Client{
		cfg:          cfg,
		http:         httpClient,
		limiter:      time.NewTicker(interval),
		log:          log,
		cookieHeader: cookieHeader,
	}, nil
}

// loadCookieHeader renders a "name=value; …" Cookie header from a browser
// cookies.json export. A missing path/file yields "" (the client still runs,
// just more exposed to WAF 403s); a malformed file is a hard error.
func loadCookieHeader(path string) (string, error) {
	if path == "" {
		return "", nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", fmt.Errorf("read cookies %s: %w", path, err)
	}
	var cookies []cookie
	if err := json.Unmarshal(data, &cookies); err != nil {
		return "", fmt.Errorf("parse cookies %s: %w", path, err)
	}
	parts := make([]string, 0, len(cookies))
	for _, c := range cookies {
		if c.Name != "" {
			parts = append(parts, c.Name+"="+c.Value)
		}
	}
	return strings.Join(parts, "; "), nil
}

// Close stops the rate-limiting ticker.
func (c *Client) Close() {
	c.limiter.Stop()
}

const maxJitter = 400 * time.Millisecond

// jitter returns a small random delay so retries from concurrent callers don't
// align into a synchronised burst.
func jitter() time.Duration {
	return time.Duration(rand.Int63n(int64(maxJitter)))
}

// sleepCtx sleeps for d unless the context is cancelled first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// FetchStatistics retrieves the period-split statistics for one event, retrying
// transient failures (403 throttle, network) with exponential backoff + jitter.
// A 404 (ErrNoStatistics) and context cancellation are returned immediately —
// neither is worth retrying. The shared rate limiter is honoured on every
// attempt, so this stays safe for concurrent callers.
func (c *Client) FetchStatistics(ctx context.Context, eventID int) (*model.StatisticsResponse, error) {
	var lastErr error
	for attempt := 0; attempt <= c.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := c.cfg.RetryBackoff*time.Duration(1<<(attempt-1)) + jitter()
			c.log.Debug("retrying after backoff", "event_id", eventID, "attempt", attempt, "backoff", backoff)
			if err := sleepCtx(ctx, backoff); err != nil {
				return nil, err
			}
		}
		resp, err := c.fetchOnce(ctx, eventID)
		if err == nil {
			return resp, nil
		}
		if errors.Is(err, ErrNoStatistics) ||
			errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		lastErr = err
	}
	return nil, lastErr
}

// fetchOnce performs a single rate-limited GET + decode (no retries).
func (c *Client) fetchOnce(ctx context.Context, eventID int) (*model.StatisticsResponse, error) {
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
	if c.cookieHeader != "" {
		req.Header.Set("Cookie", c.cookieHeader)
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

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("event %d: %w", eventID, ErrNoStatistics)
	}
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
