// electron.d.ts
export {};

export interface Credentials {
  email: string;
  password: string;
}

export interface LoginResult {
  success: boolean;
  token?: string;
  message?: string;
}

// Chat API Types
export interface ChatAPI {
  create: (params: { model: string; title: string }) => Promise<{ success: boolean; chatId?: string; title?: string; error?: string }>;
  sendMessage: (chatId: string, message: string) => Promise<{ success: boolean; response?: string; error?: string }>;
  getHistory: (chatId: string) => Promise<{ success: boolean; messages?: any[]; error?: string }>;
  getAll: () => Promise<{ success: boolean; chats?: any[]; error?: string }>;
}

// Graph API Types
export interface GraphAPI {
  createNode: (input: { id: string; label: string; category?: string }) => Promise<{ success: boolean; nodeId?: string; error?: string }>;
  createLink: (input: { source: string; target: string; relationship: string; strength?: number }) => Promise<{ success: boolean; linkId?: string; error?: string }>;
  updateNode: (nodeId: string, updates: { label?: string; category?: string }) => Promise<{ success: boolean; error?: string }>;
  updateLink: (linkId: string, updates: { relationship?: string; strength?: number }) => Promise<{ success: boolean; error?: string }>;
  deleteNode: (nodeId: string) => Promise<{ success: boolean; error?: string }>;
  deleteLink: (linkId: string) => Promise<{ success: boolean; error?: string }>;
  getData: () => Promise<{ success: boolean; data?: any; error?: string }>;
  getNodeConnections: (nodeId: string) => Promise<{ success: boolean; connections?: any[]; error?: string }>;
  getNodesByCategory: (category: string) => Promise<{ success: boolean; nodes?: any[]; error?: string }>;
  getCategories: () => Promise<{ success: boolean; categories?: string[]; error?: string }>;
  getStats: () => Promise<{ success: boolean; stats?: any; error?: string }>;
}

// Dashboard API Types
export interface DashboardAPI {
  getStats: () => Promise<{ success: boolean; stats?: { activeChats: number; notesCreated: number; aiModels: number; totalUsageHours: number }; error?: string }>;
  createNote: (params: { title: string; content: string; chatId?: string }) => Promise<{ success: boolean; noteId?: string; title?: string; error?: string }>;
  getAllNotes: () => Promise<{ success: boolean; notes?: any[]; error?: string }>;
  updateNote: (noteId: string, updates: { title?: string; content?: string }) => Promise<{ success: boolean; error?: string }>;
  deleteNote: (noteId: string) => Promise<{ success: boolean; error?: string }>;
  addAIModel: (params: { name: string; provider: string; modelId: string }) => Promise<{ success: boolean; modelId?: string; error?: string }>;
  getAllAIModels: () => Promise<{ success: boolean; models?: any[]; error?: string }>;
  toggleAIModel: (modelId: string, isActive: boolean) => Promise<{ success: boolean; error?: string }>;
}

declare global {
  interface Window {
    electronAPI: {
      ping: () => string;
      login: (credentials: Credentials) => Promise<LoginResult>;
      chat: ChatAPI;
      graph: GraphAPI;
      dashboard: DashboardAPI;
    };
  }
}