import { useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import JobsPanel from "./components/JobsPanel";
import { ConfigProvider, useConfig } from "./config";
import { AuthProvider } from "./auth";
import Logo from "./components/Logo";
import AccountMenu from "./components/AccountMenu";
import AuthDialog from "./components/AuthDialog";
import { FilmIcon, FolderIcon, PlusIcon } from "./components/Icons";
import Examples from "./pages/Examples";
import Mine from "./pages/Mine";
import Player from "./pages/Player";
import AddVideo from "./pages/AddVideo";

function Nav({ onSignIn }: { onSignIn: () => void }) {
  const { indexing_enabled } = useConfig();
  return (
    <nav className="nav">
      <div className="brand">
        <Logo />
        <span className="name">Cuepoint</span>
      </div>
      <NavLink to="/" end className="tab">
        <FilmIcon /> <span className="label">Example videos</span>
      </NavLink>
      <NavLink to="/mine" className="tab">
        <FolderIcon /> <span className="label">Your videos</span>
      </NavLink>
      <span className="spacer" />
      {indexing_enabled ? (
        <NavLink to="/add" className="tab">
          <PlusIcon /> <span className="label">Add</span>
        </NavLink>
      ) : (
        <span className="tab off" title="indexing needs a GPU and runs locally">
          <PlusIcon /> <span className="label">Add</span>
        </span>
      )}
      <AccountMenu onSignIn={onSignIn} />
    </nav>
  );
}

export default function App() {
  // The sign-in dialog lives at the top so any page can open it without a
  // route change — signing in should never lose the search you were on.
  const [authOpen, setAuthOpen] = useState(false);

  return (
    <ConfigProvider>
      <AuthProvider>
        <div className="app">
          <Nav onSignIn={() => setAuthOpen(true)} />
          <main>
            <Routes>
              <Route path="/" element={<Examples />} />
              <Route path="/mine" element={<Mine onSignIn={() => setAuthOpen(true)} />} />
              <Route path="/video/:id" element={<Player />} />
              <Route path="/add" element={<AddVideo />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
        {/* Outside <main> on purpose: it is fixed to the viewport corner and
            follows you across routes, because an index outlives a page. */}
        <JobsPanel />
        {authOpen && <AuthDialog onClose={() => setAuthOpen(false)} />}
      </AuthProvider>
    </ConfigProvider>
  );
}
