import { useState } from "react";
import "./TitleBar.css";

interface TitleBarProps {
  onSidebarToggle: (isVisible: boolean) => void;
}

export default function TitleBar({ onSidebarToggle }: TitleBarProps) {
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);

  const handleSidebarToggle = () => {
    const newVisibility = !isSidebarVisible;
    setIsSidebarVisible(newVisibility);
    onSidebarToggle(newVisibility);
  };

  return (
    <div className={`custom-titlebar ${isSidebarVisible ? 'titlebar-with-sidebar' : 'titlebar-without-sidebar'}`}>
      {/* 드래그 가능한 영역 */}
      <div className="titlebar-drag-region">
        <div className="titlebar-content">
          <div className="app-title">
            <span className="app-name">NoteHub</span>
          </div>

          <div className="titlebar-actions">
            <button
              className="titlebar-btn"
              onClick={handleSidebarToggle}
              title={isSidebarVisible ? "Hide Sidebar" : "Show Sidebar"}
            >
              <img
                src="/icons/dashboard.svg"
                alt="Sidebar"
                className="titlebar-icon"
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}