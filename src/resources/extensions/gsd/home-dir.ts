/**
 * Cross-platform home directory resolution.
 *
 * `process.env.HOME` is not set on Windows (CMD/PowerShell).
 * Falls back to USERPROFILE, then os.homedir(), then throws.
 *
 * @see https://github.com/gsd-build/gsd-2/issues/5015
 */
import { homedir } from "node:os";

export function getHomeDir(): string {
  const home = process.env.HOME || process.env.USERPROFILE || homedir();
  if (!home) {
    throw new Error(
      "Cannot resolve home directory. Set HOME or USERPROFILE environment variable.",
    );
  }
  return home;
}
