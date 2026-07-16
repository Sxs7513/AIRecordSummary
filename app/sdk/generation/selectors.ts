import type { GenerationViewState } from "./types";

export function selectGenerationText(state: GenerationViewState | undefined): string {
  return state?.blocks.map((block) => block.value).join("") ?? "";
}

export function selectGenerationIsActive(state: GenerationViewState | undefined): boolean {
  return state?.status === "queued" || state?.status === "running";
}
