"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld("electronAPI", {
    ping: () => "pong",
    login: (credentials) => electron_1.ipcRenderer.invoke("login", credentials),
});
