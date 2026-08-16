// Unit tests for CodexExecutor SSE peek: time-bounded so clients receive the
// stream live during model reasoning instead of silence until first output.
import { describe, it, expect } from "vitest";
import { CodexExecutor } from "../../open-sse/executors/codex.js";

const encoder = new TextEncoder();

function sseResponse(events) {
  let i = 0;
  const body = new ReadableStream({
    async pull(controller) {
      if (i >= events.length) { controller.close(); return; }
      const { delayMs = 0, event, payload } = events[i++];
      if (delayMs > 0) await new Promise((r) => setTimeout(r, delayMs));
      controller.enqueue(encoder.encode(`event: ${event}\ndata: ${payload}\n\n`));
    },
  });
  return new Response(body, { status: 200 });
}

async function collect(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    text += decoder.decode(value, { stream: true });
  }
  return text;
}

describe("CodexExecutor SSE peek live relay", () => {
  it("relays the stream live once the peek deadline elapses (no silence during reasoning)", async () => {
    const executor = new CodexExecutor();
    const response = sseResponse([
      { delayMs: 150, event: "response.reasoning_text.delta", payload: JSON.stringify({ type: "response.reasoning_text.delta", delta: "thinking..." }) },
      { event: "response.output_text.delta", payload: JSON.stringify({ type: "response.output_text.delta", delta: "ok" }) },
    ]);
    const started = Date.now();
    const peek = await executor._peekSseTransientError(response, { timeoutMs: 100 });
    const elapsed = Date.now() - started;
    expect(peek.matched).toBeNull();
    expect(peek.accountFallback).toBe(false);
    expect(peek.replacementBody).toBeTruthy();
    expect(elapsed).toBeLessThan(150);
    const text = await collect(peek.replacementBody);
    expect(text).toContain("reasoning_text.delta");
    expect(text).toContain("output_text.delta");
  });

  it("still detects SSE account-capacity errors inside the peek window", async () => {
    const executor = new CodexExecutor();
    const response = sseResponse([
      { event: "error", payload: JSON.stringify({ type: "error", message: "selected model is at capacity" }) },
    ]);
    const peek = await executor._peekSseTransientError(response, { timeoutMs: 2000 });
    expect(peek.accountFallback).toBe(true);
    expect(peek.replacementBody).toBeNull();
  });

  it("still stops peeking at the first user-output event", async () => {
    const executor = new CodexExecutor();
    const response = sseResponse([
      { event: "response.output_text.delta", payload: JSON.stringify({ type: "response.output_text.delta", delta: "ok" }) },
    ]);
    const peek = await executor._peekSseTransientError(response, { timeoutMs: 5000 });
    expect(peek.matched).toBeNull();
    expect(peek.replacementBody).toBeTruthy();
    const text = await collect(peek.replacementBody);
    expect(text).toContain("output_text.delta");
  });
});
