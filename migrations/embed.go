// Package migrations embeds the SQL schema migrations so a binary can apply
// them without the files being present on disk.
package migrations

import "embed"

//go:embed *.sql
var FS embed.FS
