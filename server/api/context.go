package api

import (
	"context"
	"net/http"

	"videosearch/server/store"
)

func contextWithSession(r *http.Request, id string) context.Context {
	return context.WithValue(r.Context(), sessionKey{}, id)
}

func contextWithUser(r *http.Request, u *store.User) context.Context {
	return context.WithValue(r.Context(), userKey{}, u)
}
