import { useEffect, useRef, useState } from "react";
import { useAuth } from "../auth";
import Logo from "./Logo";

type Mode = "signin" | "signup";

/**
 * Sign in or create an account.
 *
 * One dialog for both, because they differ by one field and one verb, and
 * because the common path is a visitor who already added a video and now wants
 * to keep it — which is why the dialog says what signing up actually buys them.
 */
export default function AuthDialog({ onClose }: { onClose: () => void }) {
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const emailRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    emailRef.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await (mode === "signin" ? signIn(email, password) : signUp(email, password));
      onClose();
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label="Account">
        <div className="dialog-head">
          <Logo />
          <div>
            <h2>{mode === "signin" ? "Welcome back" : "Create an account"}</h2>
            <p className="muted small">
              {mode === "signin"
                ? "Sign in to get back to your videos."
                : "Your videos stop expiring and follow you to any browser."}
            </p>
          </div>
        </div>

        <form className="stack" onSubmit={submit}>
          <label className="form-field">
            <span>Email</span>
            <input
              ref={emailRef}
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
            />
          </label>

          <label className="form-field">
            <span>Password</span>
            <input
              type="password"
              /* tells a password manager to offer a new password, not an old one */
              autoComplete={mode === "signin" ? "current-password" : "new-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "signup" ? "at least 8 characters" : "••••••••"}
              minLength={mode === "signup" ? 8 : undefined}
              required
            />
          </label>

          {error && <div className="notice error">{error}</div>}

          <button className="primary" type="submit" disabled={busy}>
            {busy ? "…" : mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>

        <div className="dialog-foot muted small">
          {mode === "signin" ? (
            <>
              No account?{" "}
              <button className="link" onClick={() => { setMode("signup"); setError(null); }}>
                Create one
              </button>
            </>
          ) : (
            <>
              Already have one?{" "}
              <button className="link" onClick={() => { setMode("signin"); setError(null); }}>
                Sign in
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
