import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import type { User } from "./api/types";

/**
 * Who is signed in, if anyone.
 *
 * An account is optional throughout: signed out, a visitor still gets an
 * anonymous session cookie and can still add and search their own videos. An
 * account only makes that corpus follow them off this browser, and stops it
 * expiring. `loading` is separate from `user === null` so the nav can avoid
 * flashing "Sign in" at someone who is already signed in.
 */
interface Auth {
  user: User | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  /** Videos moved from the anonymous session onto the account at sign-in. */
  claimed: number;
  dismissClaimed: () => void;
}

const AuthContext = createContext<Auth>({
  user: null,
  loading: true,
  signIn: async () => {},
  signUp: async () => {},
  signOut: async () => {},
  claimed: 0,
  dismissClaimed: () => {},
});

export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [claimed, setClaimed] = useState(0);

  useEffect(() => {
    api.me()
      .then((r) => setUser(r.user))
      .catch(() => setUser(null))       // signed out is not an error state
      .finally(() => setLoading(false));
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const r = await api.login(email, password);
    setUser(r.user);
    setClaimed(r.claimed_videos ?? 0);
  }, []);

  const signUp = useCallback(async (email: string, password: string) => {
    const r = await api.register(email, password);
    setUser(r.user);
    setClaimed(r.claimed_videos ?? 0);
  }, []);

  const signOut = useCallback(async () => {
    await api.logout();
    setUser(null);
    setClaimed(0);
  }, []);

  const value = useMemo<Auth>(
    () => ({ user, loading, signIn, signUp, signOut, claimed, dismissClaimed: () => setClaimed(0) }),
    [user, loading, signIn, signUp, signOut, claimed],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
