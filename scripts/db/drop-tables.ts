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

function parseArgs(argv: string[]) {
  const confirmIndex = argv.indexOf("--confirm");
  return {
    confirmedDatabase: confirmIndex >= 0 ? argv[confirmIndex + 1] : null,
    yes: argv.includes("--yes")
  };
}

function quoteIdent(value: string) {
  return `"${value.replaceAll('"', '""')}"`;
}

async function main() {
  await loadEnvFile(".env");
  const config = getDatabaseConfig();
  const args = parseArgs(process.argv.slice(2));

  if (args.confirmedDatabase !== config.database || !args.yes) {
    console.error(
      [
        "[db:drop-tables] Refusing to drop tables without explicit confirmation.",
        `Target database: ${config.database}`,
        "",
        "Run:",
        `  npm run db:drop-tables -- --confirm ${config.database} --yes`
      ].join("\n")
    );
    process.exit(1);
  }

  const pool = new pg.Pool({ connectionString: config.connectionString, max: 1 });
  try {
    const tables = await pool.query<{ table_schema: string; table_name: string }>(
      `select table_schema, table_name
         from information_schema.tables
        where table_schema = 'public'
          and table_type = 'BASE TABLE'
        order by table_name`
    );

    if (tables.rowCount === 0) {
      console.log(`[db:drop-tables] no public tables found in ${config.database}`);
      return;
    }

    const tableNames = tables.rows.map((row) => `${quoteIdent(row.table_schema)}.${quoteIdent(row.table_name)}`);
    await pool.query(`drop table ${tableNames.join(", ")} cascade`);
    console.log(`[db:drop-tables] dropped ${tables.rowCount} table(s) from ${config.database}`);
    for (const row of tables.rows) {
      console.log(`  - ${row.table_schema}.${row.table_name}`);
    }
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error("[db:drop-tables]", error);
  process.exit(1);
});
