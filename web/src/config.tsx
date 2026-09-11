import { createContext, useContext, useEffect, useState } from "react";
import { api } from "./api/client";
import type { AppConfig } from "./api/types";

/**
 * What this deployment can do. Indexing needs a GPU, so a hosted instance
 * serves an index built elsewhere and the add-video route is disabled rather
 * than hidden -- visible but off is more honest than pretending it never
 * existed.
 */
const ConfigContext = createContext<AppConfig>({
  indexing_enabled: false,
  auth_required: false,
});

export const useConfig = () => useContext(ConfigContext);

export function ConfigProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<AppConfig>({
    indexing_enabled: false,
    auth_required: false,
  });

  useEffect(() => {
    api.config().then(setConfig).catch(() => {
      /* keep the safe default: assume indexing is unavailable */
    });
  }, []);

  return <ConfigContext.Provider value={config}>{children}</ConfigContext.Provider>;
}
