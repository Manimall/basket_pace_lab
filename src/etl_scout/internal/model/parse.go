package model

import (
	"fmt"
	"strconv"
	"strings"
)

// shotSeparator divides made from attempted in Sofascore shot strings ("33/73").
const shotSeparator = "/"

// ParseShotStat parses a Sofascore made/attempted string into a ShotStat.
//
// Accepted forms (whitespace-tolerant):
//
//	"33/73"        → {Made: 33, Attempted: 73}
//	"14/18 (78%)"  → {Made: 14, Attempted: 18}   (trailing percentage ignored)
//
// It rejects empty strings, non-numeric parts, negative values, and the
// impossible case of made > attempted.
func ParseShotStat(raw string) (ShotStat, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return ShotStat{}, fmt.Errorf("empty shot string")
	}
	// Drop any trailing "(78%)" annotation, then squeeze out spaces so both
	// "14/18 (78%)" and a padded "9 / 24" normalise to "made/attempted".
	if idx := strings.IndexByte(s, '('); idx != -1 {
		s = s[:idx]
	}
	s = strings.ReplaceAll(s, " ", "")

	parts := strings.Split(s, shotSeparator)
	if len(parts) != 2 {
		return ShotStat{}, fmt.Errorf("malformed shot string %q: want made/attempted", raw)
	}

	made, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return ShotStat{}, fmt.Errorf("shot string %q: bad made value: %w", raw, err)
	}
	attempted, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil {
		return ShotStat{}, fmt.Errorf("shot string %q: bad attempted value: %w", raw, err)
	}

	if made < 0 || attempted < 0 {
		return ShotStat{}, fmt.Errorf("shot string %q: negative values not allowed", raw)
	}
	if made > attempted {
		return ShotStat{}, fmt.Errorf("shot string %q: made (%d) exceeds attempted (%d)", raw, made, attempted)
	}
	return ShotStat{Made: made, Attempted: attempted}, nil
}

// ParseCount parses a plain non-negative integer stat ("11" → 11). It rejects
// empty strings, non-numeric input, and negative values.
func ParseCount(raw string) (int, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return 0, fmt.Errorf("empty count string")
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("count string %q: not an integer: %w", raw, err)
	}
	if n < 0 {
		return 0, fmt.Errorf("count string %q: negative not allowed", raw)
	}
	return n, nil
}
