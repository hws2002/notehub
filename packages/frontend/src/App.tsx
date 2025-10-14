import { HashRouter, Routes, Route, useLocation } from "react-router-dom";
import { useState } from "react";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import TitleBar from "./components/TitleBar";

function AppContent() {
  const location = useLocation();
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const isDashboardRoute = location.pathname === "/dashboard";

  const handleSidebarToggle = (isVisible: boolean) => {
    setIsSidebarVisible(isVisible);
  };

  return (
    <>
      {/* TitleBar는 Dashboard 페이지에서만 표시 */}
      {isDashboardRoute && (
        <TitleBar onSidebarToggle={handleSidebarToggle} />
      )}

      {/* 타이틀바가 있을 때 상단 공간 확보 */}
      {isDashboardRoute && <div className="titlebar-spacer" />}

      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard isSidebarVisible={isSidebarVisible} />} />
      </Routes>
    </>
  );
}

function App() {
  return (
    <HashRouter>
      <AppContent />
    </HashRouter>
  );
}

export default App;