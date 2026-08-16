import type { ContentBlock, GenerationViewState } from "./types";

export function selectGenerationText(state: GenerationViewState | undefined): string {
  return selectContentBlocksText(state?.blocks ?? []);
}

export function selectContentBlocksText(blocks: ContentBlock[]): string {
  return blocks.map((block) => {
    if (block.type === "text") return block.value;
    if (block.type !== "AGGRE_MSG") return "";
    const primaryId = block.sub_message.message_group.primary_sub_message_id;
    const primary = block.sub_message.sub_message_list.find((item) => item.id === primaryId);
    return primary?.blocks.map((item) => item.value).join("") ?? "";
  }).join("");
}

export function selectGenerationIsActive(state: GenerationViewState | undefined): boolean {
  return state?.status === "queued" || state?.status === "running";
}
