import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { Credentials } from "@notehub/shared/types/electron";

// Add type declaration for window.electronAPI
declare global {
  interface Window {
    electronAPI: {
      login: (credentials: Credentials) => Promise<{ success: boolean; token?: string; message?: string }>;
    };
  }
}

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const result = await window.electronAPI.login({ email, password } as Credentials);
      if (result.success && result.token) {
        localStorage.setItem("jwtToken", result.token);
        setError("");
        navigate("/dashboard");
      } else {
        setError(result.message || "Login failed");
      }
    } catch {
      setError("An unexpected error occurred");
    }
  };

  return (
    <div className="d-flex justify-content-center align-items-center vh-100">
        <div className="p-4 shadow rounded" style={{ width: "100%", maxWidth: 400 }}>
        <h2 className="mb-4 text-center">Login</h2>
        <div className="mb-3">
          <label className="form-label">Email</label>
          <input
            type="email"
            className="form-control"
            value={email}
            onChange={e => setEmail(e.target.value)}
          />
        </div>
        <div className="mb-3">
          <label className="form-label">Password</label>
          <input
            type="password"
            className="form-control"
            value={password}
            onChange={e => setPassword(e.target.value)}
          />
        </div>
        {error && <div className="alert alert-danger">{error}</div>}
        <button className="btn btn-primary w-100" onClick={handleLogin}>
          Login
        </button>
      </div>
    </div>
  );
}