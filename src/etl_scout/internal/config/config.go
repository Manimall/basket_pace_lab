// Package config loads etl_scout settings from environment variables.
//
// It deliberately reuses the same DB_* variables as the Python service so a
// single .env file drives both processes. Scout-specific knobs use the
// SCOUT_/SOFASCORE_ prefixes.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Defaults centralise every tunable so nothing is hardcoded at call sites.
const (
	defaultDBHost    = "localhost"
	defaultDBPort    = "5432"
	defaultDBName    = "basket_pace"
	defaultDBUser    = "postgres"
	defaultDBPass    = "postgres"
	defaultDBSSLMode = "disable"

	defaultBaseURL = "https://api.sofascore.com/api/v1"
	// Workers default to 1: Sofascore's edge rate-limits aggressively, and at
	// 1 rps extra workers add no throughput while defeating the retry backoff
	// (other workers keep firing during a block). Raise only for gentler hosts.
	defaultWorkers        = 1
	defaultRequestsPerSec = 1
	defaultHTTPTimeout    = 15 * time.Second
	defaultCookiesPath    = "cookies.json"
	defaultMaxRetries     = 3               // per-request retries on 403/network
	defaultRetryBackoff   = 3 * time.Second // exponential base: 3s, 6s, 12s …

	defaultLeagues             = "NBA,EuroLeague"
	defaultMatchLimit          = 0   // 0 = no limit (enrich every matching match)
	defaultMaxConsecutiveFails = 5   // circuit breaker: abort after N straight fetch failures
	defaultProgressEvery       = 100 // emit a progress log every N processed matches
)

// Config is the fully-resolved runtime configuration.
type Config struct {
	DB        DBConfig
	Sofascore SofascoreConfig
	Scout     ScoutConfig
}

// DBConfig holds PostgreSQL connection parameters.
type DBConfig struct {
	Host     string
	Port     string
	Name     string
	User     string
	Password string
	SSLMode  string
}

// DSN renders a pgx-compatible keyword/value connection string.
func (c DBConfig) DSN() string {
	return fmt.Sprintf(
		"host=%s port=%s dbname=%s user=%s password=%s sslmode=%s",
		c.Host, c.Port, c.Name, c.User, c.Password, c.SSLMode,
	)
}

// SofascoreConfig controls the concurrent scraping client.
type SofascoreConfig struct {
	BaseURL        string
	Workers        int           // size of the goroutine worker pool
	RequestsPerSec int           // global rate limit (shared across workers)
	HTTPTimeout    time.Duration // per-request timeout
	CookiesPath    string        // optional cookies.json to dodge edge WAF 403s
	MaxRetries     int           // retries on 403/network before giving up on a request
	RetryBackoff   time.Duration // exponential backoff base between retries
}

// ScoutConfig controls which matches the enrichment orchestrator targets.
type ScoutConfig struct {
	Leagues             []string // leagues to enrich (matched via COALESCE(tournament,'NBA'))
	MatchLimit          int      // cap on matches per run; 0 = no limit
	MaxConsecutiveFails int      // circuit breaker: abort run after this many straight failures
	ProgressEvery       int      // log progress every N processed matches
}

// Load resolves configuration from the environment, applying defaults for any
// unset variable. It returns an error only when a present value is malformed.
func Load() (Config, error) {
	workers, err := getenvInt("SCOUT_WORKERS", defaultWorkers)
	if err != nil {
		return Config{}, err
	}
	rps, err := getenvInt("SCOUT_REQUESTS_PER_SEC", defaultRequestsPerSec)
	if err != nil {
		return Config{}, err
	}
	timeout, err := getenvDuration("SCOUT_HTTP_TIMEOUT", defaultHTTPTimeout)
	if err != nil {
		return Config{}, err
	}
	matchLimit, err := getenvInt("SCOUT_MATCH_LIMIT", defaultMatchLimit)
	if err != nil {
		return Config{}, err
	}
	maxRetries, err := getenvInt("SOFASCORE_MAX_RETRIES", defaultMaxRetries)
	if err != nil {
		return Config{}, err
	}
	retryBackoff, err := getenvDuration("SOFASCORE_RETRY_BACKOFF", defaultRetryBackoff)
	if err != nil {
		return Config{}, err
	}
	maxConsecutiveFails, err := getenvInt("SCOUT_MAX_CONSECUTIVE_FAILS", defaultMaxConsecutiveFails)
	if err != nil {
		return Config{}, err
	}
	progressEvery, err := getenvInt("SCOUT_PROGRESS_EVERY", defaultProgressEvery)
	if err != nil {
		return Config{}, err
	}
	if workers < 1 {
		return Config{}, fmt.Errorf("SCOUT_WORKERS must be >= 1, got %d", workers)
	}
	if rps < 1 {
		return Config{}, fmt.Errorf("SCOUT_REQUESTS_PER_SEC must be >= 1, got %d", rps)
	}
	if matchLimit < 0 {
		return Config{}, fmt.Errorf("SCOUT_MATCH_LIMIT must be >= 0, got %d", matchLimit)
	}
	if maxRetries < 0 {
		return Config{}, fmt.Errorf("SOFASCORE_MAX_RETRIES must be >= 0, got %d", maxRetries)
	}
	if maxConsecutiveFails < 1 {
		return Config{}, fmt.Errorf("SCOUT_MAX_CONSECUTIVE_FAILS must be >= 1, got %d", maxConsecutiveFails)
	}
	if progressEvery < 1 {
		return Config{}, fmt.Errorf("SCOUT_PROGRESS_EVERY must be >= 1, got %d", progressEvery)
	}

	leagues := getenvCSV("SCOUT_LEAGUES", defaultLeagues)
	if len(leagues) == 0 {
		return Config{}, fmt.Errorf("SCOUT_LEAGUES resolved to an empty list")
	}

	return Config{
		DB: DBConfig{
			Host:     getenv("DB_HOST", defaultDBHost),
			Port:     getenv("DB_PORT", defaultDBPort),
			Name:     getenv("DB_NAME", defaultDBName),
			User:     getenv("DB_USER", defaultDBUser),
			Password: getenv("DB_PASSWORD", defaultDBPass),
			SSLMode:  getenv("DB_SSLMODE", defaultDBSSLMode),
		},
		Sofascore: SofascoreConfig{
			BaseURL:        getenv("SOFASCORE_BASE_URL", defaultBaseURL),
			Workers:        workers,
			RequestsPerSec: rps,
			HTTPTimeout:    timeout,
			CookiesPath:    getenv("SOFASCORE_COOKIES_PATH", defaultCookiesPath),
			MaxRetries:     maxRetries,
			RetryBackoff:   retryBackoff,
		},
		Scout: ScoutConfig{
			Leagues:             leagues,
			MatchLimit:          matchLimit,
			MaxConsecutiveFails: maxConsecutiveFails,
			ProgressEvery:       progressEvery,
		},
	}, nil
}

// getenvCSV splits a comma-separated env value into trimmed, non-empty items,
// falling back to the comma-separated default when unset.
func getenvCSV(key, fallback string) []string {
	raw := getenv(key, fallback)
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func getenv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) (int, error) {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("env %s: %q is not an integer: %w", key, v, err)
	}
	return parsed, nil
}

func getenvDuration(key string, fallback time.Duration) (time.Duration, error) {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return fallback, nil
	}
	parsed, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("env %s: %q is not a duration: %w", key, v, err)
	}
	return parsed, nil
}
