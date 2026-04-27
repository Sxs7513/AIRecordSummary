import { mkdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { getAppConfig } from "../config/app-config";

const SUPPORTED_EXTENSIONS = new Set([".mp3", ".m4a", ".wav", ".mp4"]);
const SUPPORTED_MIME_TYPES = new Set(["audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a", "audio/m4a", "audio/wav", "audio/wave", "audio/x-wav"]);

export interface StoredFile {
  fileName: string;
  storagePath: string;
  mimeType: string;
  fileSizeBytes: number;
}

export function assertAudioFile(file: File): void {
  const ext = path.extname(file.name).toLowerCase();
  if (!SUPPORTED_EXTENSIONS.has(ext) && !SUPPORTED_MIME_TYPES.has(file.type)) {
    throw new Error("仅支持 mp3、m4a、wav 等基础音频格式");
  }
}

export async function saveAudioFile(file: File, folder: "recordings" | "speaker-samples"): Promise<StoredFile> {
  assertAudioFile(file);
  const storageRoot = getAppConfig().storage.localRoot;
  const safeName = path.basename(file.name).replace(/[^\w.\- ()]/g, "_");
  const relativePath = path.join(storageRoot, folder, `${randomUUID()}-${safeName}`);
  const absolutePath = path.join(process.cwd(), relativePath);
  console.log("[storage] writing audio file", {
    folder,
    fileName: file.name,
    mimeType: file.type,
    absolutePath
  });
  await mkdir(path.dirname(absolutePath), { recursive: true });
  const buffer = Buffer.from(await file.arrayBuffer());
  await writeFile(absolutePath, buffer);
  console.log("[storage] audio file written", {
    folder,
    fileName: file.name,
    storagePath: relativePath,
    fileSizeBytes: buffer.byteLength
  });
  return {
    fileName: file.name,
    storagePath: relativePath,
    mimeType: file.type || "application/octet-stream",
    fileSizeBytes: buffer.byteLength
  };
}

export function publicFileUrl(storagePath: string): string {
  const baseUrl = getAppConfig().storage.publicFileBaseUrl;
  if (baseUrl) return `${baseUrl.replace(/\/$/, "")}/${storagePath}`;
  return `/${storagePath}`;
}

export async function deleteStoredFile(storagePath: string): Promise<void> {
  const storageRoot = path.resolve(process.cwd(), getAppConfig().storage.localRoot);
  const absolutePath = path.resolve(process.cwd(), storagePath);
  if (absolutePath !== storageRoot && !absolutePath.startsWith(`${storageRoot}${path.sep}`)) {
    throw new Error("Refusing to delete file outside configured storage root");
  }

  try {
    console.log("[storage] deleting file", { storagePath, absolutePath });
    await unlink(absolutePath);
    console.log("[storage] file deleted", { storagePath });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      console.log("[storage] file already missing", { storagePath });
      return;
    }
    throw error;
  }
}
