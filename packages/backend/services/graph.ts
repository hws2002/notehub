import { db } from '../database/connection.js';
import { randomUUID } from 'crypto';
import type {
  GraphNode,
  GraphLink,
  GraphData,
  CreateNodeInput,
  CreateLinkInput
} from '../database/models/Graph.js';

// 노드 생성
export async function createNode(input: CreateNodeInput) {
  try {
    await db.run(`
      INSERT INTO graph_nodes (id, label, category)
      VALUES (?, ?, ?)
    `, [input.id, input.label, input.category || null]);

    return { success: true, nodeId: input.id };
  } catch (error) {
    console.error('Create node error:', error);
    return { success: false, error: 'Failed to create node' };
  }
}

// 링크 생성
export async function createLink(input: CreateLinkInput) {
  const linkId = randomUUID();

  try {
    await db.run(`
      INSERT INTO graph_links (id, source, target, relationship, strength)
      VALUES (?, ?, ?, ?, ?)
    `, [linkId, input.source, input.target, input.relationship, input.strength || 0.5]);

    return { success: true, linkId };
  } catch (error) {
    console.error('Create link error:', error);
    return { success: false, error: 'Failed to create link' };
  }
}

// 노드 수정
export async function updateNode(nodeId: string, updates: Partial<CreateNodeInput>) {
  try {
    const setParts = [];
    const values = [];

    if (updates.label !== undefined) {
      setParts.push('label = ?');
      values.push(updates.label);
    }
    if (updates.category !== undefined) {
      setParts.push('category = ?');
      values.push(updates.category);
    }

    setParts.push('updated_at = CURRENT_TIMESTAMP');
    values.push(nodeId);

    await db.run(`
      UPDATE graph_nodes
      SET ${setParts.join(', ')}
      WHERE id = ?
    `, values);

    return { success: true };
  } catch (error) {
    console.error('Update node error:', error);
    return { success: false, error: 'Failed to update node' };
  }
}

// 링크 수정
export async function updateLink(linkId: string, updates: Partial<CreateLinkInput>) {
  try {
    const setParts = [];
    const values = [];

    if (updates.relationship !== undefined) {
      setParts.push('relationship = ?');
      values.push(updates.relationship);
    }
    if (updates.strength !== undefined) {
      setParts.push('strength = ?');
      values.push(updates.strength);
    }

    setParts.push('updated_at = CURRENT_TIMESTAMP');
    values.push(linkId);

    await db.run(`
      UPDATE graph_links
      SET ${setParts.join(', ')}
      WHERE id = ?
    `, values);

    return { success: true };
  } catch (error) {
    console.error('Update link error:', error);
    return { success: false, error: 'Failed to update link' };
  }
}

// 노드 삭제
export async function deleteNode(nodeId: string) {
  try {
    // 관련된 링크 먼저 삭제
    await db.run(`
      DELETE FROM graph_links
      WHERE source = ? OR target = ?
    `, [nodeId, nodeId]);

    // 노드 삭제
    await db.run(`
      DELETE FROM graph_nodes
      WHERE id = ?
    `, [nodeId]);

    return { success: true };
  } catch (error) {
    console.error('Delete node error:', error);
    return { success: false, error: 'Failed to delete node' };
  }
}

// 링크 삭제
export async function deleteLink(linkId: string) {
  try {
    await db.run(`
      DELETE FROM graph_links
      WHERE id = ?
    `, [linkId]);

    return { success: true };
  } catch (error) {
    console.error('Delete link error:', error);
    return { success: false, error: 'Failed to delete link' };
  }
}

// 전체 그래프 데이터 조회 (JSON 포맷에 맞게)
export async function getGraphData(): Promise<{ success: boolean; data?: GraphData; error?: string }> {
  try {
    const nodes = await db.all(`
      SELECT id, label, category, created_at, updated_at
      FROM graph_nodes
      ORDER BY created_at DESC
    `) as GraphNode[];

    const links = await db.all(`
      SELECT id, source, target, relationship, strength, created_at, updated_at
      FROM graph_links
      ORDER BY created_at DESC
    `) as GraphLink[];

    return {
      success: true,
      data: { nodes, links }
    };
  } catch (error) {
    console.error('Get graph data error:', error);
    return { success: false, error: 'Failed to get graph data' };
  }
}

// 특정 노드와 연결된 링크들 조회
export async function getNodeConnections(nodeId: string) {
  try {
    const connections = await db.all(`
      SELECT * FROM graph_links
      WHERE source = ? OR target = ?
      ORDER BY strength DESC
    `, [nodeId, nodeId]);

    return { success: true, connections };
  } catch (error) {
    console.error('Get node connections error:', error);
    return { success: false, error: 'Failed to get node connections' };
  }
}

// 카테고리별 노드 조회
export async function getNodesByCategory(category: string) {
  try {
    const nodes = await db.all(`
      SELECT * FROM graph_nodes
      WHERE category = ?
      ORDER BY created_at DESC
    `, [category]) as GraphNode[];

    return { success: true, nodes };
  } catch (error) {
    console.error('Get nodes by category error:', error);
    return { success: false, error: 'Failed to get nodes by category' };
  }
}

// 모든 카테고리 목록 조회
export async function getAllCategories() {
  try {
    const categories = await db.all(`
      SELECT DISTINCT category
      FROM graph_nodes
      WHERE category IS NOT NULL
      ORDER BY category
    `);

    return {
      success: true,
      categories: categories.map(row => row.category)
    };
  } catch (error) {
    console.error('Get categories error:', error);
    return { success: false, error: 'Failed to get categories' };
  }
}

// 그래프 통계 조회
export async function getGraphStats() {
  try {
    const nodeCount = await db.get(`SELECT COUNT(*) as count FROM graph_nodes`);
    const linkCount = await db.get(`SELECT COUNT(*) as count FROM graph_links`);
    const categoryCount = await db.get(`
      SELECT COUNT(DISTINCT category) as count
      FROM graph_nodes
      WHERE category IS NOT NULL
    `);

    return {
      success: true,
      stats: {
        nodes: nodeCount.count,
        links: linkCount.count,
        categories: categoryCount.count
      }
    };
  } catch (error) {
    console.error('Get graph stats error:', error);
    return { success: false, error: 'Failed to get graph stats' };
  }
}