const base = {
  width: 16, height: 16, viewBox: "0 0 24 24", fill: "none",
  stroke: "currentColor", strokeWidth: 1.8,
  strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
};

export const SearchIcon = (p: { size?: number }) => (
  <svg {...base} width={p.size ?? 16} height={p.size ?? 16}>
    <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
  </svg>
);

export const FilmIcon = () => (
  <svg {...base}>
    <rect x="2.5" y="5" width="19" height="14" rx="2.5" />
    <path d="M7 5v14M17 5v14M2.5 12h19" />
  </svg>
);

export const FolderIcon = () => (
  <svg {...base}>
    <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2.5h7A1.5 1.5 0 0 1 19 10v7.5A1.5 1.5 0 0 1 17.5 19h-13A1.5 1.5 0 0 1 3 17.5Z" />
  </svg>
);

export const PlusIcon = () => (
  <svg {...base}><path d="M12 5v14M5 12h14" /></svg>
);

export const TrashIcon = () => (
  <svg {...base}>
    <path d="M4 7h16M10 7V5h4v2M6 7l1 12h10l1-12" />
  </svg>
);

export const PlayIcon = () => (
  <svg {...base} fill="currentColor" stroke="none">
    <path d="M8 5.5v13l11-6.5-11-6.5Z" />
  </svg>
);

export const ExternalIcon = () => (
  <svg {...base} width={14} height={14}>
    <path d="M14 4h6v6M20 4l-8 8M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
  </svg>
);

export const ClockIcon = () => (
  <svg {...base} width={14} height={14}>
    <circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" />
  </svg>
);

export const UserIcon = () => (
  <svg {...base}>
    <circle cx="12" cy="8" r="3.5" />
    <path d="M4.5 20a7.5 7.5 0 0 1 15 0" />
  </svg>
);

export const LayersIcon = () => (
  <svg {...base}>
    <path d="M12 3.5 3 8l9 4.5L21 8z" />
    <path d="m3 12.5 9 4.5 9-4.5" opacity=".55" />
  </svg>
);
