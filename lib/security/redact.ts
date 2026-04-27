export function redactSecrets(value: string): string {
  return value.replace(/hf_[A-Za-z0-9_=-]+/g, "hf_***");
}
