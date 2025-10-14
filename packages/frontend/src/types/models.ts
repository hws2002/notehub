// 데이터 모델 타입 정의

export interface Chat {
  id: string;
  title: string;
  model: string;
  created_at: string;
}

export interface Message {
  id: string;
  chat_id: string;
  content: string;
  role: 'user' | 'assistant';
  created_at: string;
}

export interface Note {
  id: string;
  title: string;
  content: string;
  chat_id?: string;
  created_at: string;
  updated_at: string;
}

export interface AIModel {
  id: string;
  name: string;
  provider: string;
  model_id: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface GraphNode {
  id: string;
  label: string;
  category?: string;
  created_at: string;
  updated_at: string;
}

export interface GraphLink {
  id: string;
  source: string;
  target: string;
  relationship: string;
  strength?: number;
  created_at: string;
  updated_at: string;
}

export interface DashboardStats {
  activeChats: number;
  notesCreated: number;
  aiModels: number;
  totalUsageHours: number;
}