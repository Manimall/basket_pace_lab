package model

import "testing"

func TestParseShotStat(t *testing.T) {
	cases := []struct {
		name    string
		raw     string
		want    ShotStat
		wantErr bool
	}{
		{name: "plain", raw: "32/65", want: ShotStat{Made: 32, Attempted: 65}},
		{name: "fixture field goals", raw: "33/73", want: ShotStat{Made: 33, Attempted: 73}},
		{name: "with percentage", raw: "14/18 (78%)", want: ShotStat{Made: 14, Attempted: 18}},
		{name: "padded", raw: "  9 / 24 ", want: ShotStat{Made: 9, Attempted: 24}},
		{name: "zero of zero", raw: "0/0", want: ShotStat{Made: 0, Attempted: 0}},
		{name: "empty", raw: "", wantErr: true},
		{name: "no separator", raw: "32", wantErr: true},
		{name: "non numeric", raw: "a/b", wantErr: true},
		{name: "made exceeds attempted", raw: "10/3", wantErr: true},
		{name: "negative", raw: "-1/5", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseShotStat(tc.raw)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for %q", tc.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestParseCount(t *testing.T) {
	cases := []struct {
		name    string
		raw     string
		want    int
		wantErr bool
	}{
		{name: "value", raw: "11", want: 11},
		{name: "zero", raw: "0", want: 0},
		{name: "padded", raw: " 14 ", want: 14},
		{name: "empty", raw: "", wantErr: true},
		{name: "non numeric", raw: "x", wantErr: true},
		{name: "negative", raw: "-3", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseCount(tc.raw)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for %q", tc.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}
