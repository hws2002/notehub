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

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// 그래프 생성을 위한 입력 타입
export interface CreateNodeInput {
  id: string;
  label: string;
  category?: string;
}

export interface CreateLinkInput {
  source: string;
  target: string;
  relationship: string;
  strength?: number;
}