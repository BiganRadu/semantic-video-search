/** Where a video plays back from. Never a local path: see ARCHITECTURE.md 4.1. */
export type Locator =
  | { kind: "youtube"; id: string; offset?: number }
  | { kind: "http"; url: string; offset?: number };

export type SignalName = "visual" | "caption" | "speech" | "keyword";

/** Which corpus a request addresses. The two never mix. */
export type Collection = "examples" | "mine";

/** One clip behind an assembled moment. */
export interface MomentClip {
  clip_id: number;
  start_s: number;
  end_s: number;
  score: number;
  signals: Partial<Record<SignalName, number>>;
}

/**
 * A ranked moment: a span of one video, assembled from the adjacent clips that
 * matched. `start_s`/`end_s` bound the span; `peak_s` is where playback should
 * start, since the strongest clip is rarely at the span's edge.
 */
export interface Moment {
  clip_id: number;
  video_id: string;
  start_s: number;
  end_s: number;
  peak_s?: number;
  score: number;
  /**
   * 0..1 confidence from raw similarity, not from the fusion score.
   * RRF ranks; it does not measure — inside one video its top score is the
   * same whether or not anything matched. This is the number to display.
   */
  relevance?: number;
  signals: Partial<Record<SignalName, number>>;
  locator: Locator;
  clips?: MomentClip[];
}

export interface SearchResponse {
  ok: boolean;
  error?: string;
  query: string;
  scope: "corpus" | "video";
  collection: Collection;
  video_id: string | null;
  signals: SignalName[];
  weights: Partial<Record<SignalName, number>>;
  k: number;
  corpus: { videos: number; clips: number };
  assemble?: boolean;
  /**
   * How the query was prepared: its class (which picks the weights) and the
   * phrasing sent to each index. Null when routing is off or the model did not
   * answer in time — the raw query and default weights are used then.
   */
  plan?: {
    class: string;
    confidence: number;
    queries: Partial<Record<SignalName, string>>;
    took_ms: number;
  } | null;
  coverage?: Partial<Record<SignalName | "clips", number>>;
  results: Moment[];
  took_ms: number;
}

export interface VideoSummary {
  id: string;
  locator: Locator;
  title?: string;
  source: string;
  duration_s: number;
  state: string;
  clips: number;
}

export interface ClipDetail {
  id: number;
  idx: number;
  start_s: number;
  end_s: number;
  caption: string | null;
  speech: string | null;
  objects?: string[];
  actions?: string[];
  setting?: string | null;
}

export interface TranscriptSegment {
  start_s: number;
  end_s: number;
  text: string;
}

export interface VideoDetailResponse extends Omit<VideoSummary, "clips"> {
  language?: string | null;
  pipeline?: string | null;
  /** Derived from ownership by the server. `source` is not ownership. */
  collection?: Collection;
  clips: ClipDetail[];
  transcript: TranscriptSegment[];
}

export interface User {
  id: string;
  email: string;
  created_at: string;
}

/** Every auth endpoint answers with the current user, or null when signed out. */
export interface AuthResponse {
  user: User | null;
  /** Videos added anonymously that were moved onto the account on sign-in. */
  claimed_videos?: number;
}

export interface AppConfig {
  indexing_enabled: boolean;
  auth_required: boolean;
  accounts_enabled?: boolean;
  user?: User | null;
}

/** One index job belonging to this session or account. */
export interface IndexJob {
  id: string;
  title?: string;
  /** queued while something else is indexing; only one runs at a time. */
  state: "queued" | "indexing" | "done" | "failed";
  stage?: string;
  done?: number;
  total?: number;
  /** 1-based place in the queue, present only while queued. */
  position?: number;
  error?: string;
  started: number;
}
