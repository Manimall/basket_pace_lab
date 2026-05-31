package storage

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"

	"basket_pace_lab/etl_scout/internal/model"
)

// fakeExecer captures the SQL and args of the last Exec and returns a canned err.
type fakeExecer struct {
	gotSQL  string
	gotArgs []any
	err     error
}

func (f *fakeExecer) Exec(_ context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	f.gotSQL = sql
	f.gotArgs = args
	return pgconn.CommandTag{}, f.err
}

func sampleAdvanced() model.AdvancedStats {
	return model.AdvancedStats{
		Possessions:            79.92,
		TruePace:               79.92,
		Turnovers:              10,
		ThreePointersMade:      9,
		ThreePointersAttempted: 24,
		OffensiveRating:        111.36,
		DefensiveRating:        91.34,
	}
}

func TestUpsertAdvancedMapsArgsInOrder(t *testing.T) {
	fake := &fakeExecer{}
	adv := sampleAdvanced()

	if err := upsertAdvanced(context.Background(), fake, 555, 42, adv); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(fake.gotSQL, "INSERT INTO team_match_advanced") {
		t.Errorf("SQL does not target team_match_advanced: %s", fake.gotSQL)
	}
	if !strings.Contains(fake.gotSQL, "ON CONFLICT (match_id, team_id) DO UPDATE") {
		t.Errorf("SQL is not an upsert: %s", fake.gotSQL)
	}

	want := []any{
		int64(555), int64(42), adv.Possessions, adv.TruePace, adv.Turnovers,
		adv.ThreePointersMade, adv.ThreePointersAttempted,
		adv.OffensiveRating, adv.DefensiveRating,
	}
	if len(fake.gotArgs) != len(want) {
		t.Fatalf("arg count: got %d, want %d", len(fake.gotArgs), len(want))
	}
	for i := range want {
		if fake.gotArgs[i] != want[i] {
			t.Errorf("arg[%d]: got %v (%T), want %v (%T)", i, fake.gotArgs[i], fake.gotArgs[i], want[i], want[i])
		}
	}
}

func TestUpsertAdvancedWrapsExecError(t *testing.T) {
	sentinel := errors.New("connection reset")
	fake := &fakeExecer{err: sentinel}

	err := upsertAdvanced(context.Background(), fake, 1, 2, sampleAdvanced())
	if err == nil {
		t.Fatal("expected error from failing Exec")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("error chain lost the cause: %v", err)
	}
	if !strings.Contains(err.Error(), "match 1") || !strings.Contains(err.Error(), "team 2") {
		t.Errorf("error lacks match/team context: %v", err)
	}
}
