import { useState } from "react";
import "../styles/Dashboard.css";

export default function Dashboard() {
  const token = localStorage.getItem("jwtToken");
  const [activeMenu, setActiveMenu] = useState("dashboard");

  const sidebarItems = [
    { id: "dashboard", icon: "🏠", label: "Dashboard", active: true },
    { id: "chat", icon: "💬", label: "AI Chat", active: false },
    { id: "notes", icon: "📝", label: "Notes", active: false },
    { id: "history", icon: "📋", label: "History", active: false },
    { id: "analytics", icon: "📊", label: "Analytics", active: false },
    { id: "settings", icon: "⚙️", label: "Settings", active: false },
  ];

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h4 className="sidebar-title">NoteHub</h4>
        </div>

        <nav className="sidebar-nav">
          <ul className="nav flex-column">
            {sidebarItems.map((item) => (
              <li key={item.id} className="nav-item">
                <a
                  href="#"
                  className={`nav-link ${activeMenu === item.id ? "active" : ""}`}
                  onClick={(e) => {
                    e.preventDefault();
                    setActiveMenu(item.id);
                  }}
                >
                  <span className="nav-icon">{item.icon}</span>
                  <span className="nav-text">{item.label}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="sidebar-footer">
          <div className="user-info">
            <div className="user-avatar">👤</div>
            <div className="user-details">
              <div className="user-name">User</div>
              <div className="user-email">test@example.com</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <header className="main-header">
          <div className="header-content">
            <h1 className="page-title">Dashboard</h1>
            <div className="header-actions">
              <button className="btn btn-outline-secondary btn-sm me-2">
                🔔 Notifications
              </button>
              <button className="btn btn-primary btn-sm">
                ➕ New Chat
              </button>
            </div>
          </div>
        </header>

        <main className="main-body">
          <div className="content-area">
            {activeMenu === "dashboard" && (
              <div>
                <div className="row mb-4">
                  <div className="col-md-3">
                    <div className="stat-card">
                      <div className="stat-icon">💬</div>
                      <div className="stat-content">
                        <h3>24</h3>
                        <p>Active Chats</p>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-3">
                    <div className="stat-card">
                      <div className="stat-icon">📝</div>
                      <div className="stat-content">
                        <h3>156</h3>
                        <p>Notes Created</p>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-3">
                    <div className="stat-card">
                      <div className="stat-icon">🤖</div>
                      <div className="stat-content">
                        <h3>3</h3>
                        <p>AI Models</p>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-3">
                    <div className="stat-card">
                      <div className="stat-icon">⏱️</div>
                      <div className="stat-content">
                        <h3>12h</h3>
                        <p>Total Usage</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="row">
                  <div className="col-md-8">
                    <div className="card">
                      <div className="card-header">
                        <h5>Recent Activity</h5>
                      </div>
                      <div className="card-body">
                        <p>Chat activity and note creation timeline will be displayed here.</p>
                        <div className="activity-item">
                          <span className="activity-time">2 hours ago</span>
                          <span className="activity-text">Created new chat with Claude</span>
                        </div>
                        <div className="activity-item">
                          <span className="activity-time">5 hours ago</span>
                          <span className="activity-text">Generated 3 notes from ChatGPT conversation</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="card">
                      <div className="card-header">
                        <h5>Quick Actions</h5>
                      </div>
                      <div className="card-body">
                        <div className="d-grid gap-2">
                          <button className="btn btn-primary">Start New Chat</button>
                          <button className="btn btn-outline-primary">Create Note</button>
                          <button className="btn btn-outline-secondary">View Analytics</button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeMenu !== "dashboard" && (
              <div className="text-center py-5">
                <h3>{sidebarItems.find(item => item.id === activeMenu)?.label}</h3>
                <p>This section is under development.</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}