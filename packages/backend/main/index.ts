import { app, BrowserWindow, ipcMain } from "electron";
import path from "path";
import { fileURLToPath } from "url";
import { login } from "../services/auth.js";
import { initDatabase } from "../database/connection.js";
import { createChat, sendMessage, getChatHistory, getAllChats } from "../services/chat.js";
import {
  createNode,
  createLink,
  updateNode,
  updateLink,
  deleteNode,
  deleteLink,
  getGraphData,
  getNodeConnections,
  getNodesByCategory,
  getAllCategories,
  getGraphStats
} from "../services/graph.js";
import {
  getDashboardStats,
  createNote,
  getAllNotes,
  updateNote,
  deleteNote,
  addAIModel,
  getAllAIModels,
  toggleAIModel
} from "../services/dashboard.js";
import type { Credentials } from "../types/electron";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#2f3241',
      symbolColor: '#74b1be',
      height: 50
    },
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.js"),
    },
  });

  if (process.env.NODE_ENV === "development") {
    win.loadURL("http://localhost:5173");
  } else {
    win.loadFile(path.join(__dirname, "../../../frontend/dist/index.html"));
  }
}

app.whenReady().then(async () => {
  await initDatabase();  // DB 초기화
  createWindow();
});

// Auth handlers
ipcMain.handle("login", async (_event, credentials: Credentials) => login(credentials));

// Chat handlers
ipcMain.handle("chat:create", async (_event, params) => createChat(params));
ipcMain.handle("chat:sendMessage", async (_event, chatId, message) =>
  sendMessage(chatId, message)
);
ipcMain.handle("chat:getHistory", async (_event, chatId) =>
  getChatHistory(chatId)
);
ipcMain.handle("chat:getAll", async (_event) => getAllChats());

// Graph handlers
ipcMain.handle("graph:createNode", async (_event, input) => createNode(input));
ipcMain.handle("graph:createLink", async (_event, input) => createLink(input));
ipcMain.handle("graph:updateNode", async (_event, nodeId, updates) => updateNode(nodeId, updates));
ipcMain.handle("graph:updateLink", async (_event, linkId, updates) => updateLink(linkId, updates));
ipcMain.handle("graph:deleteNode", async (_event, nodeId) => deleteNode(nodeId));
ipcMain.handle("graph:deleteLink", async (_event, linkId) => deleteLink(linkId));
ipcMain.handle("graph:getData", async (_event) => getGraphData());
ipcMain.handle("graph:getNodeConnections", async (_event, nodeId) => getNodeConnections(nodeId));
ipcMain.handle("graph:getNodesByCategory", async (_event, category) => getNodesByCategory(category));
ipcMain.handle("graph:getCategories", async (_event) => getAllCategories());
ipcMain.handle("graph:getStats", async (_event) => getGraphStats());

// Dashboard handlers
ipcMain.handle("dashboard:getStats", async (_event) => getDashboardStats());
ipcMain.handle("dashboard:createNote", async (_event, params) => createNote(params));
ipcMain.handle("dashboard:getAllNotes", async (_event) => getAllNotes());
ipcMain.handle("dashboard:updateNote", async (_event, noteId, updates) => updateNote(noteId, updates));
ipcMain.handle("dashboard:deleteNote", async (_event, noteId) => deleteNote(noteId));
ipcMain.handle("dashboard:addAIModel", async (_event, params) => addAIModel(params));
ipcMain.handle("dashboard:getAllAIModels", async (_event) => getAllAIModels());
ipcMain.handle("dashboard:toggleAIModel", async (_event, modelId, isActive) => toggleAIModel(modelId, isActive));

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});