import { readFile } from "node:fs/promises";
import path from "node:path";
import pg from "pg";
import { getAppConfig } from "../config/app-config";
import { pool } from "./pool";

declare global {
  // eslint-disable-next-line no-var
  var __aiRecordSummaryDbInitialized: Promise<void> | undefined;
}

async function runInit() {
  const config = getAppConfig().database;
  const adminPool = new pg.Pool({
    connectionString: config.adminConnectionString,
    max: 1
  });
  try {
    const existing = await adminPool.query("select 1 from pg_database where datname = $1", [config.database]);
    if (existing.rowCount === 0) {
      await adminPool.query(`create database "${config.database.replaceAll('"', '""')}"`);
    }
  } finally {
    await adminPool.end();
  }

  const schemaPath = path.join(process.cwd(), "sql", "base.sql");
  const schema = await readFile(schemaPath, "utf8");
  await pool.query(schema);
}

export function initDatabase(): Promise<void> {
  globalThis.__aiRecordSummaryDbInitialized ??= runInit();
  return globalThis.__aiRecordSummaryDbInitialized;
}
