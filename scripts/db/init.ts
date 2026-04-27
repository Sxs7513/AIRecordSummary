import { readFile } from "node:fs/promises";
import path from "node:path";
import pg from "pg";
import { getDatabaseConfig } from "../../lib/config/database";

async function loadEnvFile(fileName: string) {
  try {
    const content = await readFile(path.join(process.cwd(), fileName), "utf8");
    for (const line of content.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const index = trimmed.indexOf("=");
      if (index <= 0) continue;
      const key = trimmed.slice(0, index);
      const value = trimmed.slice(index + 1).replace(/^["']|["']$/g, "");
      process.env[key] ??= value;
    }
  } catch {
    // Optional env files are ignored.
  }
}

async function main() {
  await loadEnvFile(".env");

  const config = getDatabaseConfig();
  const adminPool = new pg.Pool({ connectionString: config.adminConnectionString, max: 1 });
  try {
    const existing = await adminPool.query("select 1 from pg_database where datname = $1", [config.database]);
    if (existing.rowCount === 0) {
      await adminPool.query(`create database "${config.database.replaceAll('"', '""')}"`);
      console.log(`[db:init] created database ${config.database}`);
    }
  } finally {
    await adminPool.end();
  }

  const schemaPath = path.join(process.cwd(), "sql", "base.sql");
  const schema = await readFile(schemaPath, "utf8");
  const pool = new pg.Pool({ connectionString: config.connectionString });
  try {
    await pool.query(schema);
    console.log("[db:init] schema is ready");
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error("[db:init]", error);
  process.exit(1);
});
