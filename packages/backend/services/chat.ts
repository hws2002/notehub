import { db } from '../database/connection.js';
import { randomUUID } from 'crypto';
import type { Chat, Message } from '../database/models/Chat.js';

export async function createChat(params: { model: string, title: string }) {
  const chatId = randomUUID();

  try {
    await db.run(`
      INSERT INTO chats (id, title, model)
      VALUES (?, ?, ?)
    `, [chatId, params.title, params.model]);

    return { success: true, chatId, title: params.title };
  } catch (error) {
    console.error('DB Error:', error);
    return { success: false, error: 'Failed to create chat' };
  }
}

export async function sendMessage(chatId: string, message: string) {
  const messageId = randomUUID();

  try {
    // 사용자 메시지 저장
    await db.run(`
      INSERT INTO messages (id, chat_id, content, role)
      VALUES (?, ?, ?, ?)
    `, [messageId, chatId, message, 'user']);

    // AI 응답 생성 (여기서 실제 AI API 호출)
    const aiResponse = await callAIAPI(message);
    const aiMessageId = randomUUID();

    // AI 응답 저장
    await db.run(`
      INSERT INTO messages (id, chat_id, content, role)
      VALUES (?, ?, ?, ?)
    `, [aiMessageId, chatId, aiResponse, 'assistant']);

    return { success: true, response: aiResponse };
  } catch (error) {
    console.error('Send message error:', error);
    return { success: false, error: 'Failed to send message' };
  }
}

export async function getChatHistory(chatId: string) {
  try {
    const messages = await db.all(`
      SELECT * FROM messages
      WHERE chat_id = ?
      ORDER BY created_at ASC
    `, [chatId]) as Message[];

    return { success: true, messages };
  } catch (error) {
    console.error('Get chat history error:', error);
    return { success: false, error: 'Failed to get chat history' };
  }
}

export async function getAllChats() {
  try {
    const chats = await db.all(`
      SELECT * FROM chats
      ORDER BY created_at DESC
    `) as Chat[];

    return { success: true, chats };
  } catch (error) {
    console.error('Get all chats error:', error);
    return { success: false, error: 'Failed to get chats' };
  }
}

async function callAIAPI(message: string): Promise<string> {
  // 실제 AI API (OpenAI, Claude 등) 호출
  // 임시로 간단한 응답 반환
  return `AI 응답: ${message}에 대한 답변입니다.`;
}