import pg from "pg";
import { getAppConfig } from "../config/app-config";

const { Pool } = pg;

declare global {
  // eslint-disable-next-line no-var
  var __aiRecordSummaryPool: pg.Pool | undefined;
}

function createPool(): pg.Pool {
  const config = getAppConfig().database;

  return new Pool({
    connectionString: config.connectionString,
    max: config.poolMax
  });
}

export const pool = globalThis.__aiRecordSummaryPool ?? createPool();

if (process.env.NODE_ENV !== "production") {
  globalThis.__aiRecordSummaryPool = pool;
}

export async function query<T>(text: string, values: unknown[] = []): Promise<T[]> {
  const result = await pool.query(text, values);
  return result.rows as T[];
}

export async function transaction<T>(fn: (client: pg.PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try {
    await client.query("begin");
    const result = await fn(client);
    await client.query("commit");
    return result;
  } catch (error) {
    await client.query("rollback");
    throw error;
  } finally {
    client.release();
  }
}
