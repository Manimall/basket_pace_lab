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

	defaultBaseURL        = "https://api.sofascore.com/api/v1"
	defaultWorkers        = 4
	defaultRequestsPerSec = 2
	defaultHTTPTimeout    = 15 * time.Second
)

// Config is the fully-resolved runtime configuration.
type Config struct {
	DB        DBConfig
	Sofascore SofascoreConfig
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
	if workers < 1 {
		return Config{}, fmt.Errorf("SCOUT_WORKERS must be >= 1, got %d", workers)
	}
	if rps < 1 {
		return Config{}, fmt.Errorf("SCOUT_REQUESTS_PER_SEC must be >= 1, got %d", rps)
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
		},
	}, nil
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
