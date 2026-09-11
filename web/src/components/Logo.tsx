/**
 * The mark: a filmstrip whose frames resolve into a marked in-point.
 *
 * The product returns a cue — a video plus a timestamp — so the logo is a
 * timeline with one frame picked out of it, not a generic play button.
 */
export default function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="cue-a" x1="4" y1="4" x2="28" y2="28" gradientUnits="userSpaceOnUse">
          <stop stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
      {/* filmstrip body */}
      <rect x="2.5" y="7.5" width="27" height="17" rx="4.5"
            stroke="url(#cue-a)" strokeWidth="2" />
      {/* perforations, fading toward the cue */}
      <rect x="6" y="11" width="2.6" height="2.6" rx="0.8" fill="url(#cue-a)" opacity=".85" />
      <rect x="6" y="18.4" width="2.6" height="2.6" rx="0.8" fill="url(#cue-a)" opacity=".85" />
      <rect x="11" y="11" width="2.6" height="2.6" rx="0.8" fill="url(#cue-a)" opacity=".45" />
      <rect x="11" y="18.4" width="2.6" height="2.6" rx="0.8" fill="url(#cue-a)" opacity=".45" />
      {/* the cue: one frame marked on the strip */}
      <path d="M19.2 12.2v7.6l6.2-3.8-6.2-3.8Z" fill="url(#cue-a)" />
      <path d="M17 5.5v21" stroke="url(#cue-a)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
