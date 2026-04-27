import { query, transaction } from "./pool";
import type { SpeakerProfile, SpeakerProfileSample, SpeakerProfileWithSamples } from "../types/models";

function rowProfile(row: Record<string, any>): SpeakerProfile {
  return {
    id: row.id,
    displayName: row.display_name,
    status: row.status,
    notes: row.notes,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString()
  };
}

function rowSample(row: Record<string, any>): SpeakerProfileSample {
  return {
    id: row.id,
    speakerProfileId: row.speaker_profile_id,
    fileName: row.file_name,
    storagePath: row.storage_path,
    mimeType: row.mime_type,
    fileSizeBytes: Number(row.file_size_bytes),
    durationSeconds: row.duration_seconds,
    status: row.status,
    errorMessage: row.error_message,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString()
  };
}

export async function listSpeakerProfiles(): Promise<SpeakerProfileWithSamples[]> {
  const [profileRows, sampleRows] = await Promise.all([
    query<Record<string, any>>("select * from speaker_profiles order by created_at desc"),
    query<Record<string, any>>("select * from speaker_profile_samples order by created_at desc")
  ]);
  const samples = sampleRows.map(rowSample);
  return profileRows.map((row) => {
    const profile = rowProfile(row);
    return {
      ...profile,
      samples: samples.filter((sample) => sample.speakerProfileId === profile.id)
    };
  });
}

export async function createSpeakerProfile(input: {
  displayName: string;
  status: "active" | "inactive";
  notes?: string | null;
}): Promise<SpeakerProfile> {
  const rows = await query<Record<string, any>>(
    `insert into speaker_profiles (display_name, status, notes)
     values ($1, $2, $3)
     returning *`,
    [input.displayName, input.status, input.notes || null]
  );
  return rowProfile(rows[0]);
}

export async function createSpeakerProfileSample(
  profileId: string,
  input: {
    fileName: string;
    storagePath: string;
    mimeType: string;
    fileSizeBytes: number;
    durationSeconds?: number | null;
  }
): Promise<SpeakerProfileSample> {
  const rows = await query<Record<string, any>>(
    `insert into speaker_profile_samples (
      speaker_profile_id, file_name, storage_path, mime_type, file_size_bytes, duration_seconds, status
    ) values ($1, $2, $3, $4, $5, $6, 'completed')
    returning *`,
    [profileId, input.fileName, input.storagePath, input.mimeType, input.fileSizeBytes, input.durationSeconds ?? null]
  );
  return rowSample(rows[0]);
}

export async function deleteSpeakerProfile(profileId: string): Promise<{ profile: SpeakerProfile; samples: SpeakerProfileSample[] } | null> {
  return transaction(async (client) => {
    const profileRows = await client.query<Record<string, any>>("select * from speaker_profiles where id = $1 for update", [profileId]);
    if (profileRows.rowCount === 0) return null;

    const sampleRows = await client.query<Record<string, any>>("select * from speaker_profile_samples where speaker_profile_id = $1", [profileId]);
    await client.query("delete from speaker_profiles where id = $1", [profileId]);

    return {
      profile: rowProfile(profileRows.rows[0]),
      samples: sampleRows.rows.map(rowSample)
    };
  });
}
