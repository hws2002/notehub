import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import type { Credentials, LoginResult } from "../../src/types/electron";

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

export async function login(credentials: Credentials): Promise<LoginResult> {
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
  return { success: true, token } as LoginResult;
}

export function verifyToken(token: string): boolean {
  try {
    jwt.verify(token, JWT_SECRET);
    return true;
  } catch {
    return false;
  }
}