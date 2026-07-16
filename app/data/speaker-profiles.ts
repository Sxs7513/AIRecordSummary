import { cookies } from "next/headers";
import type { SpeakerProfileWithSamples } from "@/app/shared/models";

const pythonApiOrigin = process.env.PYTHON_API_ORIGIN ?? "http://localhost:8000";

export async function listSpeakerProfiles(): Promise<SpeakerProfileWithSamples[]> {
  const cookie = (await cookies()).toString();
  const response = await fetch(`${pythonApiOrigin}/api/speaker-profiles`, { cache: "no-store", headers: cookie ? { cookie } : undefined });
  if (!response.ok) throw new Error(`Python speaker profile API request failed: ${response.status}`);
  const values = await response.json() as Array<Record<string, unknown>>;
  return values.map((value) => ({
    id: String(value.id), displayName: String(value.display_name), status: value.status === "inactive" ? "inactive" : "active", notes: typeof value.notes === "string" ? value.notes : null,
    createdAt: String(value.created_at), updatedAt: String(value.updated_at),
    samples: Array.isArray(value.samples) ? value.samples.map((sample) => {
      const item = sample as Record<string, unknown>;
      return { id: String(item.id), speakerProfileId: String(item.speaker_profile_id), fileName: String(item.file_name), storagePath: String(item.storage_path), mimeType: String(item.mime_type), fileSizeBytes: Number(item.file_size_bytes), durationSeconds: typeof item.duration_seconds === "number" ? item.duration_seconds : null, status: item.status as "uploaded" | "processing" | "completed" | "failed", errorMessage: typeof item.error_message === "string" ? item.error_message : null, createdAt: String(item.created_at), updatedAt: String(item.updated_at) };
    }) : []
  }));
}
