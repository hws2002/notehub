import sqlite3 from 'sqlite3';
import { app } from 'electron';
import path from 'path';

// Electron 앱 데이터 폴더에 DB 저장
const dbPath = path.join(app.getPath('userData'), 'notehub.db');

// SQLite 연결을 위한 Promise wrapper
class DatabaseWrapper {
  private db: sqlite3.Database;

  constructor(path: string) {
    this.db = new sqlite3.Database(path);
  }

  run(sql: string, params: any[] = []): Promise<sqlite3.RunResult> {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(err) {
        if (err) reject(err);
        else resolve(this);
      });
    });
  }

  get(sql: string, params: any[] = []): Promise<any> {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  all(sql: string, params: any[] = []): Promise<any[]> {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  exec(sql: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.exec(sql, (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  close(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.close((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }
}

export const db = new DatabaseWrapper(dbPath);

// 초기 테이블 생성
export async function initDatabase() {
  await db.exec(`
    CREATE TABLE IF NOT EXISTS chats (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      model TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL,
      content TEXT NOT NULL,
      role TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (chat_id) REFERENCES chats (id)
    );

    CREATE TABLE IF NOT EXISTS graph_nodes (
      id TEXT PRIMARY KEY,
      label TEXT NOT NULL,
      category TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS graph_links (
      id TEXT PRIMARY KEY,
      source TEXT NOT NULL,
      target TEXT NOT NULL,
      relationship TEXT NOT NULL,
      strength REAL DEFAULT 0.5,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (source) REFERENCES graph_nodes (id),
      FOREIGN KEY (target) REFERENCES graph_nodes (id)
    );

    CREATE TABLE IF NOT EXISTS notes (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      content TEXT NOT NULL,
      chat_id TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (chat_id) REFERENCES chats (id)
    );

    CREATE TABLE IF NOT EXISTS ai_models (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      provider TEXT NOT NULL,
      model_id TEXT NOT NULL,
      is_active BOOLEAN DEFAULT 1,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
  `);

  // 기본 AI 모델들 초기화
  await initializeDefaultAIModels();
}

// 기본 AI 모델들 초기화
async function initializeDefaultAIModels() {
  const defaultModels = [
    { id: 'gpt-4', name: 'GPT-4', provider: 'openai', model_id: 'gpt-4' },
    { id: 'claude-3', name: 'Claude 3', provider: 'anthropic', model_id: 'claude-3-sonnet' },
    { id: 'gemini-pro', name: 'Gemini Pro', provider: 'google', model_id: 'gemini-pro' }
  ];

  for (const model of defaultModels) {
    try {
      await db.run(`
        INSERT OR IGNORE INTO ai_models (id, name, provider, model_id, is_active)
        VALUES (?, ?, ?, ?, 1)
      `, [model.id, model.name, model.provider, model.model_id]);
    } catch (error) {
      console.log('AI model already exists:', model.id);
    }
  }
}

// 앱 종료 시 DB 연결 닫기
process.on('exit', () => {
  db.close();
});