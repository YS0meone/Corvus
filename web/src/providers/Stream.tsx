import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import {
  uiMessageReducer,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useQueryState } from "nuqs";
import { getApiKey } from "@/lib/api-key";
import { useThreads } from "./Thread";
import { toast } from "sonner";
import { useAuth } from "@clerk/clerk-react";

// StreamProvider only mounts inside <SignedIn>, so clerkToken will always
// be available once the token fetch resolves.

export type StateType = { messages: Message[]; ui?: UIMessage[] };

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      ui?: (UIMessage | RemoveUIMessage)[] | UIMessage | RemoveUIMessage;
      selected_paper_ids?: string[];
    };
    CustomEventType: UIMessage | RemoveUIMessage;
  }
>;

type StreamContextType = ReturnType<typeof useTypedStream>;
const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function sleep(ms = 4000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkGraphStatus(
  apiUrl: string,
  apiKey: string | null,
  clerkToken: string | null,
): Promise<boolean> {
  try {
    const headers: Record<string, string> = {};
    if (apiKey) headers["X-Api-Key"] = apiKey;
    if (clerkToken) headers["Authorization"] = `Bearer ${clerkToken}`;
    const res = await fetch(`${apiUrl}/info`, { headers });
    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiKey,
  apiUrl,
  assistantId,
  clerkToken,
}: {
  children: ReactNode;
  apiKey: string | null;
  apiUrl: string;
  assistantId: string;
  clerkToken: string | null;
}) => {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, setThreads } = useThreads();

  const defaultHeaders: Record<string, string> = {};
  if (clerkToken) defaultHeaders["Authorization"] = `Bearer ${clerkToken}`;

  const streamValue = useTypedStream({
    apiUrl,
    apiKey: apiKey ?? undefined,
    assistantId,
    threadId: threadId ?? null,
    defaultHeaders,
    onCustomEvent: (event, options) => {
      options.mutate((prev) => {
        const ui = uiMessageReducer(prev.ui ?? [], event);
        return { ...prev, ui };
      });
    },
    onThreadId: (id) => {
      setThreadId(id);
      sleep().then(() => getThreads().then(setThreads).catch(console.error));
    },
  });

  useEffect(() => {
    checkGraphStatus(apiUrl, apiKey, clerkToken).then((ok) => {
      if (!ok) {
        toast.error("Failed to connect to LangGraph server", {
          description: () => (
            <p>
              Please ensure your graph is running at <code>{apiUrl}</code> and
              your API key is correctly set (if connecting to a deployed graph).
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiKey, apiUrl, clerkToken]);

  return (
    <StreamContext.Provider value={streamValue}>
      {children}
    </StreamContext.Provider>
  );
};

// Default values for the form
const DEFAULT_API_URL = "http://localhost:2024";
const DEFAULT_ASSISTANT_ID = "agent";

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const { getToken } = useAuth();
  const [clerkToken, setClerkToken] = useState<string | null>(null);

  useEffect(() => {
    const fetchToken = async () => {
      // skipCache: true forces Clerk to mint a new JWT every call.
      // Without it, getToken() returns the same cached token until expiry,
      // so clerkToken state never changes, useStream's client is never
      // recreated, and requests fail with 403 "Signature has expired".
      const token = await getToken({ skipCache: true });
      setClerkToken(token);
    };
    fetchToken();
    // Tokens expire after 60 s; refresh every 45 s to stay well inside that.
    const interval = setInterval(fetchToken, 45 * 1000);
    return () => clearInterval(interval);
  }, [getToken]);

  const envApiUrl: string | undefined = import.meta.env.VITE_API_URL;
  const envAssistantId: string | undefined = import.meta.env.VITE_ASSISTANT_ID;
  const envApiKey: string | undefined = import.meta.env.VITE_LANGSMITH_API_KEY;

  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId] = useQueryState("assistantId", {
    defaultValue: envAssistantId || "",
  });

  const [apiKey] = useState(() => {
    const storedKey = getApiKey();
    return storedKey || envApiKey || "";
  });

  const finalApiUrl = apiUrl || envApiUrl;
  const finalAssistantId = assistantId || envAssistantId;

  // Wait until the Clerk token is fetched before mounting StreamSession,
  // so useStream's internal client is always initialized with the auth header.
  if (!clerkToken) return null;

  return (
    <StreamSession
      apiKey={apiKey || null}
      apiUrl={finalApiUrl || DEFAULT_API_URL}
      assistantId={finalAssistantId || DEFAULT_ASSISTANT_ID}
      clerkToken={clerkToken}
    >
      {children}
    </StreamSession>
  );
};

export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
