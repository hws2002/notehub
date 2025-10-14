// API 응답 타입 정의
import type { Chat, Message, Note, AIModel, GraphNode, GraphLink, DashboardStats } from './models';

// 기본 API 응답 타입
export interface BaseResponse {
  success: boolean;
  error?: string;
}

// Chat API 응답 타입들
export interface CreateChatResponse extends BaseResponse {
  chatId?: string;
  title?: string;
}

export interface SendMessageResponse extends BaseResponse {
  response?: string;
}

export interface GetChatHistoryResponse extends BaseResponse {
  messages?: Message[];
}

export interface GetAllChatsResponse extends BaseResponse {
  chats?: Chat[];
}

// Dashboard API 응답 타입들
export interface GetDashboardStatsResponse extends BaseResponse {
  stats?: DashboardStats;
}

export interface CreateNoteResponse extends BaseResponse {
  noteId?: string;
  title?: string;
}

export interface GetAllNotesResponse extends BaseResponse {
  notes?: Note[];
}

export interface GetAllAIModelsResponse extends BaseResponse {
  models?: AIModel[];
}

// Graph API 응답 타입들
export interface CreateNodeResponse extends BaseResponse {
  nodeId?: string;
}

export interface CreateLinkResponse extends BaseResponse {
  linkId?: string;
}

export interface GetGraphDataResponse extends BaseResponse {
  data?: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
}

export interface GetNodeConnectionsResponse extends BaseResponse {
  connections?: GraphLink[];
}

export interface GetNodesByCategoryResponse extends BaseResponse {
  nodes?: GraphNode[];
}

export interface GetCategoriesResponse extends BaseResponse {
  categories?: string[];
}

export interface GetGraphStatsResponse extends BaseResponse {
  stats?: {
    nodes: number;
    links: number;
    categories: number;
  };
}