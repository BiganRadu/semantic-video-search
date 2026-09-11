import { Link } from "react-router-dom";
import CorpusPage from "../components/CorpusPage";
import { PlusIcon, UserIcon } from "../components/Icons";
import { useConfig } from "../config";
import { useAuth } from "../auth";

/**
 * The visitor's own corpus.
 *
 * Signed out it belongs to the browser: the server hands out an anonymous
 * session cookie and anything added under it lives for a week. Signed in it
 * belongs to the account, stops expiring, and follows them to another browser.
 * The page is the same either way — an account changes how long the corpus
 * lasts and where it can be reached from, not what it can do.
 */
export default function Mine({ onSignIn }: { onSignIn: () => void }) {
  const { indexing_enabled } = useConfig();
  const { user, loading, claimed, dismissClaimed } = useAuth();

  const addButton = indexing_enabled ? (
    <Link to="/add">
      <button className="primary"><PlusIcon /> Add a video</button>
    </Link>
  ) : null;

  const blurb = user
    ? `Signed in as ${user.email}. These videos are kept on your account.`
    : "Videos you add, kept to this browser for a week. Sign in to keep them for good.";

  return (
    <>
      {claimed > 0 && (
        <div className="notice ok claim" role="status">
          <b>{claimed === 1 ? "1 video moved" : `${claimed} videos moved`} to your account.</b>{" "}
          <span className="muted">They no longer expire.</span>
          <button className="link" onClick={dismissClaimed}>dismiss</button>
        </div>
      )}

      {!user && !loading && (
        <div className="notice subtle signin-hint">
          <UserIcon />
          <span className="grow">
            These videos live in this browser and expire after a week.
          </span>
          <button className="ghost" onClick={onSignIn}>Sign in to keep them</button>
        </div>
      )}

      {/*
        Keyed by identity: signing in or out swaps the whole corpus, so the page
        must refetch rather than keep showing the previous owner's videos.
      */}
      <CorpusPage
        key={user?.id ?? "anon"}
        collection="mine"
        title="Your videos"
        blurb={blurb}
        canManage={indexing_enabled}
        onAdd={addButton}
        emptyState={
          <div className="notice">
            <b>You haven’t added any videos yet.</b>
            <div className="muted" style={{ marginTop: 8, lineHeight: 1.7 }}>
              {indexing_enabled ? (
                <>
                  Add one and it becomes searchable the same way the examples are.
                  <br />
                  {user
                    ? "They are kept on your account until you delete them."
                    : "They stay tied to this browser for a week, then expire."}
                </>
              ) : (
                <>
                  Indexing needs a GPU, so it runs locally. This instance only
                  serves an index that was built elsewhere.
                </>
              )}
            </div>
            {indexing_enabled && (
              <div style={{ marginTop: 16 }}>
                <Link to="/add">
                  <button className="primary"><PlusIcon /> Add a video</button>
                </Link>
              </div>
            )}
          </div>
        }
      />
    </>
  );
}
