export function publicFileUrl(storagePath: string): string {
  return `/uploads/${storagePath.split("/").map(encodeURIComponent).join("/")}`;
}
