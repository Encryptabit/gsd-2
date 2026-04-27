// gsd-2 / merge-self-branch-guard.test.ts — regression for #5024
//
// mergeMilestoneToMain() must fail closed when the resolved integration
// branch is the same ref as the milestone branch. Stale or corrupt
// integration metadata (e.g. integrationBranch recorded as "milestone/<MID>")
// would otherwise let the squash merge resolve to a self-merge: the post-
// merge no-op safety check (#1792) compares main vs milestone and finds an
// empty diff (because they're the same ref), so the helper returns success
// for work that never landed on a distinct integration branch.

import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  existsSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execFileSync } from "node:child_process";

import { mergeMilestoneToMain } from "../auto-worktree.ts";
import { _resetServiceCache } from "../worktree.ts";
import { _clearGsdRootCache } from "../paths.ts";

function git(args: string[], cwd: string): string {
  return execFileSync("git", args, { cwd, stdio: ["ignore", "pipe", "pipe"], encoding: "utf-8" }).trim();
}

function createTempRepo(): string {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "merge-self-guard-")));
  git(["init"], dir);
  git(["config", "user.email", "test@test.com"], dir);
  git(["config", "user.name", "Test"], dir);
  writeFileSync(join(dir, "README.md"), "# test\n");
  git(["add", "."], dir);
  git(["commit", "-m", "init"], dir);
  git(["branch", "-M", "main"], dir);
  return dir;
}

function assertSelfMergeRefIsRejected(recordedIntegrationBranch: string): void {
  const savedCwd = process.cwd();
  let tempDir = "";

  // Isolate from user's global preferences so prefs.main_branch can't
  // override the corrupt-metadata path under test.
  const originalHome = process.env.HOME;
  const fakeHome = realpathSync(mkdtempSync(join(tmpdir(), "gsd-fake-home-")));
  process.env.HOME = fakeHome;
  _clearGsdRootCache();
  _resetServiceCache();

  try {
    tempDir = createTempRepo();

    // Plant corrupt integration metadata: integrationBranch points at the
    // milestone branch itself. Commit it so mergeMilestoneToMain's
    // autoCommitDirtyState pre-step has nothing to capture and the
    // postcondition (no new commits) cleanly reflects the guard.
    const msDir = join(tempDir, ".gsd", "milestones", "M001");
    mkdirSync(msDir, { recursive: true });
    writeFileSync(
      join(msDir, "M001-META.json"),
      JSON.stringify({ integrationBranch: recordedIntegrationBranch }),
    );
    git(["add", "."], tempDir);
    git(["commit", "-m", "chore: plant corrupt M001 meta"], tempDir);

    // Create the milestone branch ref so any pre-guard branch operations
    // wouldn't fail for unrelated reasons.
    git(["branch", "milestone/M001"], tempDir);

    const mainHeadBefore = git(["rev-parse", "main"], tempDir);
    const milestoneHeadBefore = git(["rev-parse", "milestone/M001"], tempDir);

    process.chdir(tempDir);

    assert.throws(
      () => mergeMilestoneToMain(tempDir, "M001", ""),
      (err: unknown) => {
        assert.ok(err instanceof Error, "expected an Error to be thrown");
        assert.match(
          err.message,
          /self-merge|same ref/i,
          "error message should explain the self-merge refusal",
        );
        return true;
      },
    );

    // Postcondition: neither branch should have been advanced by a merge
    // commit. The guard fires before checkout/merge, so both refs must be
    // unchanged from their pre-call state.
    const mainHeadAfter = git(["rev-parse", "main"], tempDir);
    const milestoneHeadAfter = git(["rev-parse", "milestone/M001"], tempDir);
    assert.equal(mainHeadAfter, mainHeadBefore, "main must not have advanced");
    assert.equal(
      milestoneHeadAfter,
      milestoneHeadBefore,
      "milestone branch must not have advanced",
    );
  } finally {
    process.chdir(savedCwd);
    process.env.HOME = originalHome;
    _clearGsdRootCache();
    _resetServiceCache();
    if (tempDir && existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true });
    }
    rmSync(fakeHome, { recursive: true, force: true });
  }
}

test("mergeMilestoneToMain refuses exact milestone branch self-merge metadata (#5024)", () => {
  assertSelfMergeRefIsRejected("milestone/M001");
});

test("mergeMilestoneToMain refuses refs/heads milestone branch self-merge metadata (#5024)", () => {
  assertSelfMergeRefIsRejected("refs/heads/milestone/M001");
});
