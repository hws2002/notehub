import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
// JWT secret key (실제 서비스에서는 env나 secure vault 사용)
const JWT_SECRET = "supersecretkey";
// 예제용 DB: 실제 서비스에서는 DB 사용
const usersDB = [
    {
        email: "test@example.com",
        // 비밀번호 해시: bcrypt.hashSync("1234", 10)
        passwordHash: bcrypt.hashSync("1234", 10),
    },
];
export async function login(credentials) {
    const { email, password } = credentials;
    const user = usersDB.find(u => u.email === email);
    if (!user) {
        return { success: false, message: "User not found" };
    }
    const passwordValid = await bcrypt.compare(password, user.passwordHash);
    if (!passwordValid) {
        return { success: false, message: "Invalid password" };
    }
    const token = jwt.sign({ email }, JWT_SECRET, { expiresIn: "1h" });
    return { success: true, token };
}
export function verifyToken(token) {
    try {
        jwt.verify(token, JWT_SECRET);
        return true;
    }
    catch {
        return false;
    }
}
