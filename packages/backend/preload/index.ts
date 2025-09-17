import { contextBridge, ipcRenderer } from "electron";
import type { Credentials, LoginResult } from "../types/electron";

contextBridge.exposeInMainWorld("electronAPI", {
  ping: () => "pong",
  login: (credentials: Credentials): Promise<LoginResult> =>
    ipcRenderer.invoke("login", credentials),
});