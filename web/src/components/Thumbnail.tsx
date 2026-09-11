import { useState } from "react";
import type { Locator } from "../api/types";
import { thumbnailUrl } from "../api/time";
import { FilmIcon } from "./Icons";

/**
 * A still for a video card.
 *
 * YouTube stills come straight from its CDN — nothing is stored on our side,
 * which is the same rule the rest of the system follows for video bytes.
 * Anything else gets a generated placeholder: deriving a frame from an
 * arbitrary URL would mean downloading and decoding it.
 *
 * A YouTube locator can rot — the video gets deleted or set private — and when
 * it does, YouTube answers with a 120x90 grey image rather than a 404. So a
 * successful load is not proof of a usable still; the width has to be checked.
 * See ARCHITECTURE.md on locator rot.
 */
export default function Thumbnail({
  locator, alt, seed,
}: {
  locator: Locator;
  alt: string;
  /** Stable input for the placeholder's colour, so a card looks the same every visit. */
  seed: string;
}) {
  const src = thumbnailUrl(locator);
  const [failed, setFailed] = useState(false);

  if (!src || failed) return <Placeholder seed={seed} unavailable={failed} />;

  return (
    <div className="thumb">
      <img
        src={src}
        alt={alt}
        loading="lazy"
        decoding="async"
        // Do not leak the page the user is on to a third-party CDN.
        referrerPolicy="no-referrer"
        onError={() => setFailed(true)}
        onLoad={(e) => {
          // YouTube's "no such video" image is 120x90. A real mqdefault is 320
          // wide, so anything narrower means the locator has rotted.
          if (e.currentTarget.naturalWidth > 0 && e.currentTarget.naturalWidth < 200) {
            setFailed(true);
          }
        }}
      />
    </div>
  );
}

/**
 * Stands in for a still we cannot get. Coloured from the id so a wall of cards
 * reads as distinct items rather than a wall of identical grey boxes, and so
 * the same video keeps the same colour between visits.
 */
function Placeholder({ seed, unavailable }: { seed: string; unavailable?: boolean }) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  const hue = h % 360;

  return (
    <div
      className="thumb placeholder"
      style={{
        background:
          `linear-gradient(135deg, hsl(${hue} 42% 26%), hsl(${(hue + 48) % 360} 38% 16%))`,
      }}
      aria-label={unavailable ? "preview unavailable" : "no preview"}
    >
      <FilmIcon />
      {unavailable && <span className="badge">unavailable</span>}
    </div>
  );
}
