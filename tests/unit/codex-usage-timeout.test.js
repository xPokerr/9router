// Unit tests for the Codex usage-fetch deadline: a stalled upstream must fail
// with a clear timeout error instead of occupying the quota route indefinitely.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const mocks = vi.hoisted(() => ({ proxyAwareFetch: vi.fn() }));
vi.mock("../../open-sse/utils/proxyFetch.js", () => ({ proxyAwareFetch: mocks.proxyAwareFetch }));

import { getCodexUsage, CODEX_USAGE_TIMEOUT_MS } from "../../open-sse/services/usage/codex.js";

beforeEach(() => {
  mocks.proxyAwareFetch.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Codex usage fetch deadline", () => {
  it("aborts the usage fetch after the timeout", async () => {
    // Simulate a fetch that honors the abort signal and never resolves otherwise.
    mocks.proxyAwareFetch.mockImplementation(async (url, opts) => {
      return await new Promise((_, reject) => {
        opts.signal?.addEventListener("abort", () => {
          reject(new DOMException("The operation was aborted due to timeout", "TimeoutError"));
        });
      });
    });

    vi.useFakeTimers();
    const promise = getCodexUsage("tok");
    // Attach the rejection handler before the fake timers fire, so the abort
    // rejection is never observed as unhandled.
    const assertion = expect(promise).rejects.toThrow(/aborted due to timeout/);
    await vi.advanceTimersByTimeAsync(CODEX_USAGE_TIMEOUT_MS + 50);
    await assertion;
  });

  it("returns parsed usage on success (no regression)", async () => {
    mocks.proxyAwareFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        plan_type: "plus",
        rate_limit: { limit_reached: false, primary_window: { limit: 100, used: 10 } },
      }),
    });

    const usage = await getCodexUsage("tok");
    expect(usage.plan).toBe("plus");
    expect(usage.limitReached).toBe(false);
  });
});
