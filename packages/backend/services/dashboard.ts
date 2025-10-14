import { db } from '../database/connection.js';
import { randomUUID } from 'crypto';
import type { Note, AIModel, DashboardStats } from '../database/models/Note.js';

// 대시보드 통계 조회
export async function getDashboardStats(): Promise<{ success: boolean; stats?: DashboardStats; error?: string }> {
  try {
    // 1. Active Chats 수
    const chatCountResult = await db.get(`SELECT COUNT(*) as count FROM chats`);
    const activeChats = chatCountResult.count;

    // 2. Notes Created 수
    const notesCountResult = await db.get(`SELECT COUNT(*) as count FROM notes`);
    const notesCreated = notesCountResult.count;

    // 3. AI Models 수 (활성화된 것만)
    const aiModelsCountResult = await db.get(`SELECT COUNT(*) as count FROM ai_models WHERE is_active = 1`);
    const aiModels = aiModelsCountResult.count;

    // 4. Total Usage Hours (임시로 메시지 수 기반 계산)
    const messagesCountResult = await db.get(`SELECT COUNT(*) as count FROM messages`);
    const totalUsageHours = Math.round((messagesCountResult.count * 0.5) * 10) / 10; // 메시지 당 0.5분 가정

    return {
      success: true,
      stats: {
        activeChats,
        notesCreated,
        aiModels,
        totalUsageHours
      }
    };
  } catch (error) {
    console.error('Get dashboard stats error:', error);
    return { success: false, error: 'Failed to get dashboard stats' };
  }
}

// 노트 생성
export async function createNote(params: { title: string, content: string, chatId?: string }) {
  const noteId = randomUUID();

  try {
    await db.run(`
      INSERT INTO notes (id, title, content, chat_id)
      VALUES (?, ?, ?, ?)
    `, [noteId, params.title, params.content, params.chatId || null]);

    return { success: true, noteId, title: params.title };
  } catch (error) {
    console.error('Create note error:', error);
    return { success: false, error: 'Failed to create note' };
  }
}

// 모든 노트 조회
export async function getAllNotes() {
  try {
    const notes = await db.all(`
      SELECT * FROM notes
      ORDER BY created_at DESC
    `) as Note[];

    return { success: true, notes };
  } catch (error) {
    console.error('Get all notes error:', error);
    return { success: false, error: 'Failed to get notes' };
  }
}

// 노트 수정
export async function updateNote(noteId: string, updates: { title?: string, content?: string }) {
  try {
    const setParts = [];
    const values = [];

    if (updates.title !== undefined) {
      setParts.push('title = ?');
      values.push(updates.title);
    }
    if (updates.content !== undefined) {
      setParts.push('content = ?');
      values.push(updates.content);
    }

    setParts.push('updated_at = CURRENT_TIMESTAMP');
    values.push(noteId);

    await db.run(`
      UPDATE notes
      SET ${setParts.join(', ')}
      WHERE id = ?
    `, values);

    return { success: true };
  } catch (error) {
    console.error('Update note error:', error);
    return { success: false, error: 'Failed to update note' };
  }
}

// 노트 삭제
export async function deleteNote(noteId: string) {
  try {
    await db.run(`DELETE FROM notes WHERE id = ?`, [noteId]);
    return { success: true };
  } catch (error) {
    console.error('Delete note error:', error);
    return { success: false, error: 'Failed to delete note' };
  }
}

// AI 모델 추가
export async function addAIModel(params: { name: string, provider: string, modelId: string }) {
  const id = randomUUID();

  try {
    await db.run(`
      INSERT INTO ai_models (id, name, provider, model_id, is_active)
      VALUES (?, ?, ?, ?, 1)
    `, [id, params.name, params.provider, params.modelId]);

    return { success: true, modelId: id };
  } catch (error) {
    console.error('Add AI model error:', error);
    return { success: false, error: 'Failed to add AI model' };
  }
}

// 모든 AI 모델 조회
export async function getAllAIModels() {
  try {
    const models = await db.all(`
      SELECT * FROM ai_models
      ORDER BY created_at ASC
    `) as AIModel[];

    return { success: true, models };
  } catch (error) {
    console.error('Get AI models error:', error);
    return { success: false, error: 'Failed to get AI models' };
  }
}

// AI 모델 활성화/비활성화
export async function toggleAIModel(modelId: string, isActive: boolean) {
  try {
    await db.run(`
      UPDATE ai_models
      SET is_active = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `, [isActive ? 1 : 0, modelId]);

    return { success: true };
  } catch (error) {
    console.error('Toggle AI model error:', error);
    return { success: false, error: 'Failed to toggle AI model' };
  }
}