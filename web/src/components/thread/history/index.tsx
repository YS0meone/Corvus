import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useRef, useState } from "react";
import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ChevronUp, LogOut, MessageSquare, PanelRightOpen, SquarePen } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";
import { useUser, useClerk } from "@clerk/clerk-react";

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function ThreadListSkeleton() {
  return (
    <div className="flex-1 px-2 py-2 space-y-1">
      {[80, 60, 72, 55, 68].map((w, i) => (
        <div key={i} className="px-3 py-2 rounded-md">
          <div
            className="h-3.5 rounded bg-muted-foreground/15 animate-pulse mb-1.5"
            style={{ width: `${w}%` }}
          />
          <div className="h-2.5 rounded bg-muted-foreground/10 animate-pulse w-16" />
        </div>
      ))}
    </div>
  );
}

function ThreadList({
  threads,
  threadsLoading,
  onThreadClick,
}: {
  threads: Thread[];
  threadsLoading: boolean;
  onThreadClick?: () => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");

  if (threadsLoading) {
    return <ThreadListSkeleton />;
  }

  if (threads.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center flex-1 gap-2 text-muted-foreground">
        <MessageSquare className="size-7 opacity-25" />
        <p className="text-sm">No conversations yet</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-200 [&::-webkit-scrollbar-track]:bg-transparent px-2 py-2">
      {threads.map((t) => {
        let itemText = t.thread_id;
        if (
          typeof t.values === "object" &&
          t.values &&
          "messages" in t.values &&
          Array.isArray(t.values.messages) &&
          t.values.messages?.length > 0
        ) {
          const firstMessage = t.values.messages[0];
          itemText = getContentString(firstMessage.content);
        }
        const isActive = t.thread_id === threadId;
        const dateStr = t.updated_at ? formatDate(t.updated_at) : "";

        return (
          <button
            key={t.thread_id}
            onClick={(e) => {
              e.preventDefault();
              onThreadClick?.();
              if (t.thread_id === threadId) return;
              setThreadId(t.thread_id);
            }}
            className={cn(
              "w-full text-left rounded-md px-3 py-2 mb-0.5 transition-colors",
              isActive
                ? "bg-accent text-accent-foreground"
                : "hover:bg-accent/60 text-foreground",
            )}
          >
            <p className="truncate text-sm leading-snug">{itemText}</p>
            {dateStr && (
              <p className="text-xs text-muted-foreground mt-0.5">{dateStr}</p>
            )}
          </button>
        );
      })}
    </div>
  );
}

function UserSection() {
  const { user } = useUser();
  const { signOut } = useClerk();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  if (!user) return null;

  const initials = [user.firstName, user.lastName]
    .filter(Boolean)
    .map((n) => n![0].toUpperCase())
    .join("") || user.emailAddresses[0]?.emailAddress[0].toUpperCase() || "?";

  const displayName =
    [user.firstName, user.lastName].filter(Boolean).join(" ") ||
    user.emailAddresses[0]?.emailAddress ||
    "Account";

  const email = user.emailAddresses[0]?.emailAddress ?? "";

  return (
    <div ref={containerRef} className="relative border-t border-border/60">
      {/* Popover — opens upward */}
      {open && (
        <div className="absolute bottom-full left-2 right-2 mb-1 rounded-lg border border-border bg-background shadow-lg overflow-hidden z-50">
          {/* Email */}
          <div className="px-3 py-2.5">
            <p className="text-xs text-muted-foreground truncate">{email}</p>
          </div>
          <div className="h-px bg-border/60 mx-1" />
          {/* Sign out */}
          <button
            onClick={() => signOut({ redirectUrl: "/sign-in" })}
            className="flex w-full items-center gap-2.5 px-3 py-2.5 text-sm text-foreground hover:bg-accent/60 transition-colors"
          >
            <LogOut className="size-4 text-muted-foreground" strokeWidth={1.75} />
            Sign out
          </button>
        </div>
      )}

      {/* Trigger */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 hover:bg-accent/60 transition-colors"
      >
        {/* Avatar */}
        <div className="size-7 rounded-full bg-gray-900 text-white flex items-center justify-center text-xs font-semibold shrink-0">
          {initials}
        </div>

        {/* Name */}
        <span className="flex-1 truncate text-sm text-left font-medium text-foreground">
          {displayName}
        </span>

        <ChevronUp
          className={cn(
            "size-4 text-muted-foreground transition-transform shrink-0",
            open ? "rotate-180" : "",
          )}
          strokeWidth={1.75}
        />
      </button>
    </div>
  );
}

function SidebarContent({
  threads,
  threadsLoading,
  onClose,
  onNewThread,
  onThreadClick,
}: {
  threads: Thread[];
  threadsLoading: boolean;
  onClose: () => void;
  onNewThread: () => void;
  onThreadClick?: () => void;
}) {
  return (
    <>
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border/60">
        <span className="text-sm font-semibold tracking-tight px-1">
          Conversations
        </span>
        <div className="flex items-center gap-0.5">
          <Button
            size="icon"
            variant="ghost"
            className="size-8"
            onClick={onNewThread}
            title="New conversation"
          >
            <SquarePen className="size-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="size-8"
            onClick={onClose}
            title="Close sidebar"
          >
            <PanelRightOpen className="size-4" />
          </Button>
        </div>
      </div>
      <ThreadList threads={threads} threadsLoading={threadsLoading} onThreadClick={onThreadClick} />
      <UserSection />
    </>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [, setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } = useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, []);

  const handleClose = () => setChatHistoryOpen(false);
  const handleNewThread = () => setThreadId(null);

  return (
    <>
      {/* Desktop: fixed sidebar */}
      <div className="hidden lg:flex flex-col border-r border-border h-screen w-[260px] shrink-0 bg-muted/20">
        <SidebarContent
          threads={threads}
          threadsLoading={threadsLoading}
          onClose={handleClose}
          onNewThread={handleNewThread}
        />
      </div>

      {/* Mobile: sheet */}
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent side="left" className="flex flex-col p-0 w-[260px]">
            <SheetHeader className="px-3 py-2.5 border-b border-border/60">
              <SheetTitle className="text-sm font-semibold px-1">
                Conversations
              </SheetTitle>
            </SheetHeader>
            <ThreadList threads={threads} threadsLoading={threadsLoading} onThreadClick={handleClose} />
            <UserSection />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
