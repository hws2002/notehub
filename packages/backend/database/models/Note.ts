export interface Note {
  id: string;
  title: string;
  content: string;
  chat_id?: string; // 채팅에서 생성된 노트라면 연결
  created_at: string;
  updated_at: string;
}

export interface AIModel {
  id: string;
  name: string;
  provider: string; // 'openai', 'anthropic', 'google'
  model_id: string; // 'gpt-4', 'claude-3', 'gemini-pro'
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface DashboardStats {
  activeChats: number;
  notesCreated: number;
  aiModels: number;
  totalUsageHours: number;
}