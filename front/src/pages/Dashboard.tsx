export default function Dashboard() {
  const token = localStorage.getItem("jwtToken");
  return (
    <div className="container mt-5">
      <h1>Dashboard</h1>
      <p>JWT Token: {token}</p>
    </div>
  );
}