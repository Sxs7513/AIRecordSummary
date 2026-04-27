export interface DatabaseConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
  adminDatabase: string;
  ssl: boolean;
  poolMax: number;
  connectionString: string;
  adminConnectionString: string;
}

function buildUrl(input: {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
  ssl: boolean;
}) {
  const url = new URL("postgres://localhost");
  url.hostname = input.host;
  url.port = String(input.port);
  url.username = input.user;
  url.password = input.password;
  url.pathname = `/${input.database}`;
  if (input.ssl) url.searchParams.set("sslmode", "require");
  return url.toString();
}

export function getDatabaseConfig(env: NodeJS.ProcessEnv = process.env): DatabaseConfig {
  const explicitUrl = env.DATABASE_URL ? new URL(env.DATABASE_URL) : null;
  const host = env.DB_HOST || explicitUrl?.hostname || "localhost";
  const port = Number(env.DB_PORT || explicitUrl?.port || 5432);
  const user = env.DB_USER || decodeURIComponent(explicitUrl?.username || "postgres");
  const password = env.DB_PASSWORD || decodeURIComponent(explicitUrl?.password || "postgres");
  const database = env.DB_NAME || explicitUrl?.pathname.replace(/^\//, "") || "ai_record_summary";
  const adminDatabase = env.DB_ADMIN_DATABASE || "postgres";
  const ssl = (env.DB_SSL || explicitUrl?.searchParams.get("sslmode") || "").toLowerCase() === "require";
  const poolMax = Number(env.PG_POOL_MAX || 10);

  return {
    host,
    port,
    user,
    password,
    database,
    adminDatabase,
    ssl,
    poolMax,
    connectionString: env.DATABASE_URL || buildUrl({ host, port, user, password, database, ssl }),
    adminConnectionString: buildUrl({ host, port, user, password, database: adminDatabase, ssl })
  };
}
