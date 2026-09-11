import { useEffect, useRef, useState } from "react";
import { useAuth } from "../auth";
import { UserIcon } from "./Icons";

/**
 * The account control in the nav: an avatar and menu when signed in, a plain
 * "Sign in" button when not.
 *
 * It never blocks anything. Signing in is an upgrade — your videos stop
 * expiring and follow you to another browser — not a gate in front of the app.
 */
export default function AccountMenu({ onSignIn }: { onSignIn: () => void }) {
  const { user, loading, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  // Hold the space while the first /api/auth/me is in flight rather than
  // flashing "Sign in" at someone who is already signed in.
  if (loading) return <span className="tab off avatar-skeleton" aria-hidden="true" />;

  if (!user) {
    return (
      <button className="ghost signin" onClick={onSignIn}>
        <UserIcon /> <span className="label">Sign in</span>
      </button>
    );
  }

  const initial = user.email.trim().charAt(0).toUpperCase() || "?";

  return (
    <div className="account" ref={ref}>
      <button
        className="avatar-button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={user.email}
      >
        <span className="avatar">{initial}</span>
        <span className="email label">{user.email}</span>
      </button>

      {open && (
        <div className="menu" role="menu">
          <div className="menu-head">
            <span className="avatar large">{initial}</span>
            <div className="grow" style={{ minWidth: 0 }}>
              <div className="truncate">{user.email}</div>
              <div className="muted small">Your videos are kept on this account</div>
            </div>
          </div>
          <button
            className="menu-item"
            role="menuitem"
            onClick={async () => {
              setOpen(false);
              await signOut();
            }}
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
