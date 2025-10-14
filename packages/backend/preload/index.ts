import { contextBridge, ipcRenderer } from "electron";
import type { Credentials, LoginResult } from "../types/electron";

contextBridge.exposeInMainWorld("electronAPI", {
  ping: () => "pong",
  login: (credentials: Credentials): Promise<LoginResult> =>
    ipcRenderer.invoke("login", credentials),

  // Chat API
  chat: {
    create: (params: { model: string; title: string }) =>
      ipcRenderer.invoke("chat:create", params),
    sendMessage: (chatId: string, message: string) =>
      ipcRenderer.invoke("chat:sendMessage", chatId, message),
    getHistory: (chatId: string) =>
      ipcRenderer.invoke("chat:getHistory", chatId),
    getAll: () =>
      ipcRenderer.invoke("chat:getAll"),
  },

  // Graph API
  graph: {
    createNode: (input: { id: string; label: string; category?: string }) =>
      ipcRenderer.invoke("graph:createNode", input),
    createLink: (input: { source: string; target: string; relationship: string; strength?: number }) =>
      ipcRenderer.invoke("graph:createLink", input),
    updateNode: (nodeId: string, updates: { label?: string; category?: string }) =>
      ipcRenderer.invoke("graph:updateNode", nodeId, updates),
    updateLink: (linkId: string, updates: { relationship?: string; strength?: number }) =>
      ipcRenderer.invoke("graph:updateLink", linkId, updates),
    deleteNode: (nodeId: string) =>
      ipcRenderer.invoke("graph:deleteNode", nodeId),
    deleteLink: (linkId: string) =>
      ipcRenderer.invoke("graph:deleteLink", linkId),
    getData: () =>
      ipcRenderer.invoke("graph:getData"),
    getNodeConnections: (nodeId: string) =>
      ipcRenderer.invoke("graph:getNodeConnections", nodeId),
    getNodesByCategory: (category: string) =>
      ipcRenderer.invoke("graph:getNodesByCategory", category),
    getCategories: () =>
      ipcRenderer.invoke("graph:getCategories"),
    getStats: () =>
      ipcRenderer.invoke("graph:getStats"),
  },

  // Dashboard API
  dashboard: {
    getStats: () =>
      ipcRenderer.invoke("dashboard:getStats"),
    createNote: (params: { title: string; content: string; chatId?: string }) =>
      ipcRenderer.invoke("dashboard:createNote", params),
    getAllNotes: () =>
      ipcRenderer.invoke("dashboard:getAllNotes"),
    updateNote: (noteId: string, updates: { title?: string; content?: string }) =>
      ipcRenderer.invoke("dashboard:updateNote", noteId, updates),
    deleteNote: (noteId: string) =>
      ipcRenderer.invoke("dashboard:deleteNote", noteId),
    addAIModel: (params: { name: string; provider: string; modelId: string }) =>
      ipcRenderer.invoke("dashboard:addAIModel", params),
    getAllAIModels: () =>
      ipcRenderer.invoke("dashboard:getAllAIModels"),
    toggleAIModel: (modelId: string, isActive: boolean) =>
      ipcRenderer.invoke("dashboard:toggleAIModel", modelId, isActive),
  },
});