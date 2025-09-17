import { app, BrowserWindow, ipcMain } from "electron";
import path from "path";
import { fileURLToPath } from "url";
import { login } from "./services/auth.js";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
function createWindow() {
    const win = new BrowserWindow({
        width: 1000,
        height: 700,
        webPreferences: {
            preload: path.join(__dirname, "./preload.js"),
        },
    });
    if (process.env.NODE_ENV === "development") {
        win.loadURL("http://localhost:5173");
    }
    else {
        win.loadFile(path.join(__dirname, "../../dist/index.html"));
    }
}
app.whenReady().then(createWindow);
ipcMain.handle("login", async (_event, credentials) => login(credentials));
app.on("window-all-closed", () => {
    if (process.platform !== "darwin")
        app.quit();
});
