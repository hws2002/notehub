// electron.d.ts
export {};

declare global {
  interface Window {
    electronAPI: {
      ping: () => string;
      login: (credentials: Credentials) => Promise<LoginResult>;
    };
  }
}

// types/electron.ts
export interface Credentials {
  email: string;
  password: string;
}

export interface LoginResult {
  success: boolean;
  token?: string;
  message?: string;
}