import type { RagRoute } from "./route-schema";

export interface RagRouterProvider {
  routeQuery(query: string): Promise<RagRoute>;
}
