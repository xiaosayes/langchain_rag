import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import * as api from "../api/client";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

interface AuthContextValue {
  status: AuthStatus;
  csrfToken: string | null;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [csrfToken, setCsrfToken] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    api
      .me()
      .then((session) => {
        if (!active) return;
        setCsrfToken(session.csrf_token);
        setStatus("authenticated");
      })
      .catch(() => {
        if (!active) return;
        setCsrfToken(null);
        setStatus("unauthenticated");
      });
    return () => {
      active = false;
    };
  }, []);

  const signIn = useCallback(async (username: string, password: string) => {
    const session = await api.login(username, password);
    setCsrfToken(session.csrf_token);
    setStatus("authenticated");
  }, []);

  const signOut = useCallback(async () => {
    if (csrfToken) await api.logout(csrfToken);
    setCsrfToken(null);
    setStatus("unauthenticated");
  }, [csrfToken]);

  const value = useMemo(
    () => ({ status, csrfToken, signIn, signOut }),
    [status, csrfToken, signIn, signOut],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}
