package config

import (
	"strings"
	"testing"
	"time"
)

func TestLoadAppliesOverrides(t *testing.T) {
	t.Setenv("DB_HOST", "db.internal")
	t.Setenv("DB_PORT", "5434")
	t.Setenv("DB_NAME", "basket_pace")
	t.Setenv("SCOUT_WORKERS", "8")
	t.Setenv("SCOUT_REQUESTS_PER_SEC", "5")
	t.Setenv("SCOUT_HTTP_TIMEOUT", "30s")
	t.Setenv("SOFASCORE_BASE_URL", "https://example.test/api")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: unexpected error: %v", err)
	}
	if cfg.DB.Host != "db.internal" || cfg.DB.Port != "5434" {
		t.Errorf("DB host/port not applied: %+v", cfg.DB)
	}
	if cfg.Sofascore.Workers != 8 || cfg.Sofascore.RequestsPerSec != 5 {
		t.Errorf("sofascore concurrency not applied: %+v", cfg.Sofascore)
	}
	if cfg.Sofascore.HTTPTimeout != 30*time.Second {
		t.Errorf("timeout not parsed: got %v", cfg.Sofascore.HTTPTimeout)
	}
	if cfg.Sofascore.BaseURL != "https://example.test/api" {
		t.Errorf("base url not applied: %q", cfg.Sofascore.BaseURL)
	}
}

func TestLoadRejectsMalformedInt(t *testing.T) {
	t.Setenv("SCOUT_WORKERS", "not-a-number")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for non-integer SCOUT_WORKERS")
	}
}

func TestLoadRejectsNonPositiveWorkers(t *testing.T) {
	t.Setenv("SCOUT_WORKERS", "0")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for SCOUT_WORKERS=0")
	}
}

func TestDSNContainsAllParts(t *testing.T) {
	dsn := DBConfig{
		Host: "h", Port: "1", Name: "n", User: "u", Password: "p", SSLMode: "disable",
	}.DSN()
	for _, part := range []string{"host=h", "port=1", "dbname=n", "user=u", "password=p", "sslmode=disable"} {
		if !strings.Contains(dsn, part) {
			t.Errorf("DSN missing %q: %s", part, dsn)
		}
	}
}
