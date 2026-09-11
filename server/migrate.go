package main

// Schema migrations, as a subcommand of the server binary:
//
//	server migrate up          apply all pending migrations (default)
//	server migrate down        roll back the most recent migration
//	server migrate status      show what has been applied
//	server migrate reset       roll everything back
//
// The SQL lives in migrations/ and is embedded, so the binary carries its own
// schema history and does not need the repo checked out beside it.

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"

	"videosearch/migrations"
)

// runMigrations is `server migrate <cmd>`.
func runMigrations(args []string) {
	log.SetFlags(0)

	// A private flag set: the server's own flags are not this command's, and
	// parsing them here would reject `server migrate up` outright.
	fs := flag.NewFlagSet("migrate", flag.ExitOnError)
	dsn := fs.String("dsn", os.Getenv("DATABASE_URL"), "postgres connection string")
	_ = fs.Parse(args)

	if *dsn == "" {
		log.Fatal("migrate: no DSN (pass -dsn or set DATABASE_URL)")
	}

	cmd := "up"
	if rest := fs.Args(); len(rest) > 0 {
		cmd = rest[0]
	}

	db, err := sql.Open("pgx", *dsn)
	if err != nil {
		log.Fatalf("migrate: open: %v", err)
	}
	defer db.Close()

	if err := db.PingContext(context.Background()); err != nil {
		log.Fatalf("migrate: connect: %v", err)
	}

	goose.SetBaseFS(migrations.FS)
	goose.SetLogger(log.Default())
	if err := goose.SetDialect("postgres"); err != nil {
		log.Fatalf("migrate: dialect: %v", err)
	}

	if err := goosePlan(db, cmd); err != nil {
		log.Fatalf("migrate: %v", err)
	}
}

func goosePlan(db *sql.DB, cmd string) error {
	switch cmd {
	case "up":
		return goose.Up(db, ".")
	case "down":
		return goose.Down(db, ".")
	case "status":
		return goose.Status(db, ".")
	case "reset":
		return goose.Reset(db, ".")
	default:
		return fmt.Errorf("unknown command %q (want up, down, status or reset)", cmd)
	}
}
