// Unit tests for stream-stall fallback wiring: the controller must surface the
// stall error so the chat handler can lock the stalled account for that model.
import { describe, it, expect } from "vitest";
import { createStreamController, isStreamStallTimeout } from "../../open-sse/utils/streamHandler.js";

describe("stream stall fallback", () => {
  it("controller onError receives the stall error", () => {
    const seen = [];
    const ctrl = createStreamController({ onError: (e) => seen.push(e), log: {}, provider: "codex", model: "gpt-5.6-luna" });
    ctrl.handleError(new Error("stream stall timeout"));
    expect(seen).toHaveLength(1);
    expect(isStreamStallTimeout(seen[0])).toBe(true);
  });

  it("distinguishes stall timeouts from other stream errors", () => {
    expect(isStreamStallTimeout(new Error("stream stall timeout"))).toBe(true);
    expect(isStreamStallTimeout(new Error("boom"))).toBe(false);
    expect(isStreamStallTimeout(null)).toBe(false);
  });
});
