package main

import "testing"

func TestParseEventIDs(t *testing.T) {
	cases := []struct {
		name    string
		raw     string
		want    []int
		wantErr bool
	}{
		{name: "empty", raw: "", want: nil},
		{name: "whitespace only", raw: "  ", want: nil},
		{name: "single", raw: "123", want: []int{123}},
		{name: "multiple with spaces", raw: " 1, 2 ,3 ", want: []int{1, 2, 3}},
		{name: "trailing comma", raw: "1,2,", want: []int{1, 2}},
		{name: "invalid", raw: "1,abc", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseEventIDs(tc.raw)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for %q", tc.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("len mismatch: got %v want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("index %d: got %d want %d", i, got[i], tc.want[i])
				}
			}
		})
	}
}
