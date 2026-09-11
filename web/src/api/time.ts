import type { Locator } from "./types";

export function mmss(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * Playback position in the ORIGINAL video.
 *
 * Every timestamp the API returns is relative to the indexed excerpt.
 * locator.offset is added exactly once, here, and nowhere else in the system.
 */
export function playbackAt(locator: Locator, startS: number): number {
  return Math.floor((locator.offset ?? 0) + startS);
}

export function watchUrl(locator: Locator, startS: number): string | null {
  const at = playbackAt(locator, startS);
  if (locator.kind === "youtube") return `https://www.youtube.com/watch?v=${locator.id}&t=${at}s`;
  if (locator.kind === "http") return `${locator.url}#t=${at}`;
  return null;
}

export function embedUrl(locator: Locator, startS: number): string | null {
  const at = playbackAt(locator, startS);
  if (locator.kind === "youtube") {
    return `https://www.youtube-nocookie.com/embed/${locator.id}?start=${at}&rel=0`;
  }
  return null;
}

/**
 * A still for this video, or null if we have no way to get one.
 *
 * YouTube serves these from its own CDN, so a thumbnail costs us nothing to
 * store — which matters, because not storing video assets is the whole shape of
 * this project. `mqdefault` is a true 16:9 320x180; `hqdefault` is 480x360 with
 * black bars baked in, which would letterbox every card.
 *
 * Nothing else has a thumbnail: an arbitrary http video would have to be
 * decoded to get a frame, and that needs ffmpeg and the bytes. Those fall back
 * to a generated placeholder instead.
 */
export function thumbnailUrl(locator: Locator): string | null {
  if (locator.kind === "youtube") return `https://i.ytimg.com/vi/${locator.id}/mqdefault.jpg`;
  return null;
}
