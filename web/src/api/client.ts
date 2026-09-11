import type {
  AppConfig, AuthResponse, Collection, SearchResponse, SignalName,
  VideoDetailResponse, VideoSummary,
} from "./types";

async function get<T>(path: string): Promise<T> {
  // Cookies carry the session -- anonymous or signed in -- so every request
  // must send them.
  const res = await fetch(path, { credentials: "same-origin" });
  return unwrap<T>(res);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  return unwrap<T>(res);
}

async function unwrap<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok || (body as { ok?: boolean })?.ok === false) {
    throw new Error((body as { error?: string })?.error ?? `${res.status} ${res.statusText}`);
  }
  return body as T;
}

export const api = {
  config: () => get<AppConfig>("/api/config"),

  // Accounts are optional. Signed out, `me` returns { user: null } with a 200 —
  // not being logged in is a normal state, not an error.
  me: () => get<AuthResponse>("/api/auth/me"),
  register: (email: string, password: string) =>
    post<AuthResponse>("/api/auth/register", { email, password }),
  login: (email: string, password: string) =>
    post<AuthResponse>("/api/auth/login", { email, password }),
  logout: () => post<AuthResponse>("/api/auth/logout"),

  search: (opts: {
    q: string;
    collection: Collection;
    k?: number;
    signals?: SignalName[];
    scope?: "corpus" | "video";
    videoId?: string;
  }) => {
    // k is deliberately not defaulted here. The server picks it per scope --
    // a within-video search wants a few timestamps, a corpus search wants
    // breadth -- so there is one place that number lives.
    const params = new URLSearchParams({ q: opts.q, collection: opts.collection });
    if (opts.k) params.set("k", String(opts.k));
    if (opts.signals?.length) params.set("signals", opts.signals.join(","));
    if (opts.scope) params.set("scope", opts.scope);
    if (opts.videoId) params.set("video_id", opts.videoId);
    return get<SearchResponse>(`/api/search?${params}`);
  },

  videos: (collection: Collection) =>
    get<{ videos: VideoSummary[]; collection: Collection }>(
      `/api/videos?collection=${collection}`,
    ),

  video: (id: string) => get<VideoDetailResponse>(`/api/videos/${encodeURIComponent(id)}`),

  deleteVideo: async (id: string) => {
    const res = await fetch(`/api/videos/${encodeURIComponent(id)}`, {
      method: "DELETE",
      credentials: "same-origin",
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error ?? "delete failed");
  },
};
