/**
 * Tests for getHomeDir() — cross-platform home directory resolution.
 *
 * @see https://github.com/gsd-build/gsd-2/issues/5015
 */
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { homedir } from "node:os";

describe("getHomeDir", () => {
  let savedHome: string | undefined;
  let savedUserProfile: string | undefined;
  let getHomeDir: () => string;

  beforeEach(async () => {
    savedHome = process.env.HOME;
    savedUserProfile = process.env.USERPROFILE;
    const mod = await import("../home-dir.ts");
    getHomeDir = mod.getHomeDir;
  });

  afterEach(() => {
    if (savedHome !== undefined) {
      process.env.HOME = savedHome;
    } else {
      delete process.env.HOME;
    }
    if (savedUserProfile !== undefined) {
      process.env.USERPROFILE = savedUserProfile;
    } else {
      delete process.env.USERPROFILE;
    }
  });

  it("returns HOME when set", () => {
    process.env.HOME = "/test/home";
    delete process.env.USERPROFILE;
    assert.equal(getHomeDir(), "/test/home");
  });

  it("falls back to USERPROFILE when HOME is unset", () => {
    delete process.env.HOME;
    process.env.USERPROFILE = String.raw`C:\Users\test`;
    assert.equal(getHomeDir(), String.raw`C:\Users\test`);
  });

  it("falls back to os.homedir() when both HOME and USERPROFILE are unset", () => {
    delete process.env.HOME;
    delete process.env.USERPROFILE;
    assert.equal(getHomeDir(), homedir());
  });
});
