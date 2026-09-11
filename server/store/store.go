// Package store is everything that touches Postgres: schema types, reads and
// writes. It is the one package both the indexing and the search path use, and
// it must never pull in either of them.
//
//	store.go     connection and helpers
//	write.go     videos, clips, vectors, transcripts
//	users.go     accounts and sessions
//	locator.go   where a video plays back from
package store

import (
	"context"
	"fmt"
	"strconv"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct{ Pool *pgxpool.Pool }

func Open(ctx context.Context, dsn string) (*Store, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("store: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("store: connect: %w", err)
	}
	return &Store{Pool: pool}, nil
}

func (s *Store) Close() { s.Pool.Close() }

// Vec renders a float slice in pgvector's text form.
//
// The text form is used rather than the binary protocol so that no type OIDs
// need registering: "$1::vector" and "$1::halfvec" both accept it, and the
// same helper serves both column types.
func Vec(v []float32) string {
	if v == nil {
		return ""
	}
	b := make([]byte, 0, len(v)*10+2)
	b = append(b, '[')
	for i, f := range v {
		if i > 0 {
			b = append(b, ',')
		}
		b = strconv.AppendFloat(b, float64(f), 'g', -1, 32)
	}
	return string(append(b, ']'))
}
