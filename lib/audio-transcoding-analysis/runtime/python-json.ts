import { spawn } from "node:child_process";
import { redactSecrets } from "../../security/redact.ts";

export interface AudioProgressEvent {
  stage: string;
  message: string;
  percent: number;
}

export async function runPythonJson<T>(options: {
  pythonBin: string;
  args: string[];
  env: NodeJS.ProcessEnv;
  logPrefix: string;
  onProgress?: (progress: AudioProgressEvent) => void;
}): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const child = spawn(options.pythonBin, options.args, {
      env: options.env,
      cwd: process.cwd()
    });

    let stdout = "";
    let stderr = "";
    let stderrBuffer = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      stderrBuffer += text;
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() ?? "";
      for (const line of lines) {
        const marker = "PROGRESS_JSON:";
        if (line.startsWith(marker)) {
          try {
            options.onProgress?.(JSON.parse(line.slice(marker.length)) as AudioProgressEvent);
          } catch (error) {
            console.error(`${options.logPrefix} invalid progress event`, { line, error });
          }
        } else if (line.trim()) {
          console.log(`${options.logPrefix} stderr`, redactSecrets(line));
        }
      }
    });

    child.on("error", reject);
    child.on("exit", (code) => {
      if (code !== 0) {
        reject(new Error(redactSecrets(`Command failed: ${options.pythonBin} ${options.args.join(" ")}\n${stderr}`)));
        return;
      }
      try {
        resolve(JSON.parse(stdout) as T);
      } catch (error) {
        console.error(`${options.logPrefix} invalid JSON`, { stdoutPreview: stdout.slice(0, 500) });
        reject(error);
      }
    });
  });
}
