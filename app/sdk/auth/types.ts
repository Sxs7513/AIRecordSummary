export type WorkspaceMembership = { id: string; name: string; role: "owner" | "admin" | "member" };

export type AuthUser = {
  id: string;
  email: string;
  display_name: string;
  current_workspace_id: string;
  memberships: WorkspaceMembership[];
};
