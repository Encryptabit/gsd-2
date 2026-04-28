/**
 * Status predicates for GSD state-machine guards.
 *
 * The DB stores status as free-form strings. Three values indicate
 * "closed": "complete" (canonical), "done" (legacy / alias), and
 * "skipped" (user-directed skip via rethink or backtrack).
 * Every inline `status === "complete" || status === "done"` should
 * use isClosedStatus() / isCompletedStatus() instead.
 */

/** Returns true when a milestone, slice, or task status indicates closure. */
export function isClosedStatus(status: string): boolean {
  return status === "complete" || status === "done" || status === "skipped";
}

/**
 * Returns true when status indicates the unit completed successfully —
 * i.e. closed but NOT skipped. Use this when the distinction matters
 * (e.g. complete-task's idempotent path treats "complete"/"done" as a
 * benign duplicate but "skipped" as an explicit user opt-out that needs
 * gsd_task_reopen first).
 */
export function isCompletedStatus(status: string): boolean {
  return status === "complete" || status === "done";
}

/** Returns true when a slice status indicates it was deferred by a decision. */
export function isDeferredStatus(status: string): boolean {
  return status === "deferred";
}

/**
 * Returns true when a slice should be skipped during active-slice selection.
 * This includes both closed (complete/done) and deferred slices.
 */
export function isInactiveStatus(status: string): boolean {
  return isClosedStatus(status) || isDeferredStatus(status);
}
