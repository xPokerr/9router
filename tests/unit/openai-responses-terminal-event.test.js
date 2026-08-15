import { describe, expect, it, vi } from "vitest";

import { FORMATS } from "../../open-sse/translator/formats.js";
import { createPassthroughStreamWithLogger, createSSETransformStreamWithLogger } from "../../open-sse/utils/stream.js";

async function runTransform(input, chunkSize = null, targetFormat = FORMATS.OPENAI_RESPONSES, sourceFormat = FORMATS.OPENAI_RESPONSES, passthrough = false, onStreamError = null) {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(input);
  const stream = new ReadableStream({
    start(controller) {
      if (!chunkSize) {
        controller.enqueue(bytes);
      } else {
        for (let offset = 0; offset < bytes.length; offset += chunkSize) {
          controller.enqueue(bytes.slice(offset, offset + chunkSize));
        }
      }
      controller.close();
    },
  });

  const transform = passthrough
    ? createPassthroughStreamWithLogger("codex", null, "gpt-5.5", null, null, null, null, targetFormat, sourceFormat, onStreamError)
    : createSSETransformStreamWithLogger(targetFormat, sourceFormat, "codex", null, null, "gpt-5.5", null, null, null, null, null, onStreamError);
  const output = stream.pipeThrough(transform);

  const reader = output.getReader();
  const decoder = new TextDecoder();
  let text = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    text += decoder.decode(value, { stream: true });
  }

  text += decoder.decode();
  return text;
}

describe("OpenAI Responses streaming termination", () => {
  it("translates completed text to Chat Completions with finish_reason", async () => {
    const output = await runTransform([
      `event: response.created`,
      `data: ${JSON.stringify({ type: "response.created", response: { id: "resp_test", status: "in_progress" } })}`,
      "",
      `event: response.output_text.delta`,
      `data: ${JSON.stringify({ type: "response.output_text.delta", delta: "ok" })}`,
      "",
      `event: response.completed`,
      `data: ${JSON.stringify({ type: "response.completed", response: { id: "resp_test", status: "completed" } })}`,
      "",
    ].join("\n"), 7, FORMATS.OPENAI_RESPONSES, FORMATS.OPENAI);

    expect(output).toContain('"content":"ok"');
    expect(output).toContain('"finish_reason":"stop"');
    expect(output.match(/data: \[DONE\]/g)).toHaveLength(1);
  });

  it("includes inclusive cache usage on normal Chat completion", async () => {
    const output = await runTransform([
      `data: ${JSON.stringify({ id: "chat_cache", choices: [{ index: 0, delta: { content: "ok" }, finish_reason: null }] })}`,
      "",
      `data: ${JSON.stringify({ id: "chat_cache", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13, prompt_tokens_details: { cached_tokens: 7, cache_creation_tokens: 2 }, completion_tokens_details: { reasoning_tokens: 1 } } })}`,
      "",
    ].join("\n"), null, FORMATS.OPENAI, FORMATS.OPENAI_RESPONSES);

    const completed = JSON.parse(output.match(/event: response.completed\ndata: (.+)\n\n/)[1]);
    expect(completed.response.usage).toEqual({
      input_tokens: 10,
      output_tokens: 3,
      total_tokens: 13,
      input_tokens_details: { cached_tokens: 7, cache_creation_tokens: 2 },
      output_tokens_details: { reasoning_tokens: 1 },
    });
  });

  it("translates a completed tool call with finish_reason tool_calls", async () => {
    const output = await runTransform([
      `event: response.output_item.added`,
      `data: ${JSON.stringify({ type: "response.output_item.added", item: { type: "function_call", call_id: "call_1", name: "test_ping" } })}`,
      "",
      `event: response.function_call_arguments.delta`,
      `data: ${JSON.stringify({ type: "response.function_call_arguments.delta", delta: "{}" })}`,
      "",
      `event: response.output_item.done`,
      `data: ${JSON.stringify({ type: "response.output_item.done", item: { type: "function_call", call_id: "call_1", name: "test_ping", arguments: "{}" } })}`,
      "",
      `event: response.completed`,
      `data: ${JSON.stringify({ type: "response.completed", response: { id: "resp_test", status: "completed" } })}`,
      "",
    ].join("\n"), null, FORMATS.OPENAI_RESPONSES, FORMATS.OPENAI);

    expect(output).toContain('"name":"test_ping"');
    expect(output).toContain('"finish_reason":"tool_calls"');
    expect(output.match(/data: \[DONE\]/g)).toHaveLength(1);
  });

  it("emits incomplete for raw DONE without a Chat finish reason", async () => {
    const output = await runTransform([
      `data: ${JSON.stringify({ id: "chat_raw_done", choices: [{ index: 0, delta: { content: "partial" }, finish_reason: null }] })}`,
      "",
      "data: [DONE]\n\n",
    ].join("\n"), null, FORMATS.OPENAI, FORMATS.OPENAI_RESPONSES);

    expect(output).toContain("event: response.incomplete");
    expect(output).not.toContain("event: response.completed");
    expect(output).toContain('"code":"protocol_incomplete"');
  });

  it("emits failed for a top-level Chat error", async () => {
    const output = await runTransform(`data: ${JSON.stringify({ error: { code: "rate_limit", message: "slow down" } })}\n\n`, null, FORMATS.OPENAI, FORMATS.OPENAI_RESPONSES);

    expect(output.match(/event: response.failed/g)).toHaveLength(1);
    expect(output).toContain('"code":"rate_limit"');
    expect(output).toContain('"message":"slow down"');
    expect(output).not.toContain("event: response.completed");
  });

  it("completes on length or content_filter finish reasons", async () => {
    for (const reason of ["length", "content_filter"]) {
      const output = await runTransform(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "partial" }, finish_reason: reason }] })}\n\n`, null, FORMATS.OPENAI, FORMATS.OPENAI_RESPONSES);
      expect(output).toContain("event: response.completed");
      expect(output).not.toContain("event: response.incomplete");
      expect(output).not.toContain("event: response.failed");
    }
  });

  it("does not close a partial tool call before incomplete EOF", async () => {
    const output = await runTransform(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call_partial", function: { name: "ping", arguments: "{\\\"x\\\":" } }] }, finish_reason: null }] })}\n\n`, null, FORMATS.OPENAI, FORMATS.OPENAI_RESPONSES);

    expect(output).toContain("event: response.incomplete");
    expect(output).not.toContain("event: response.function_call_arguments.done");
    expect(output).not.toContain("event: response.output_item.done");
  });

  it("does not synthesize finish_reason or DONE after premature EOF", async () => {
    const output = await runTransform([
      `event: response.created`,
      `data: ${JSON.stringify({ type: "response.created", response: { id: "resp_test", status: "in_progress" } })}`,
      "",
      `event: response.output_text.delta`,
      `data: ${JSON.stringify({ type: "response.output_text.delta", delta: "partial" })}`,
      "",
    ].join("\n"), null, FORMATS.OPENAI_RESPONSES, FORMATS.OPENAI);

    expect(output).toContain('"code":"protocol_incomplete"');
    expect(output).not.toContain('"finish_reason":"stop"');
    expect(output).not.toContain('"finish_reason":"tool_calls"');
    expect(output).not.toContain("data: [DONE]");
  });

  it("reports one protocol error for premature EOF", async () => {
    const onStreamError = vi.fn();
    await runTransform([
      `event: response.created`,
      `data: ${JSON.stringify({ type: "response.created", response: { id: "resp_test", status: "in_progress" } })}`,
      "",
    ].join("\n"), null, FORMATS.OPENAI_RESPONSES, FORMATS.OPENAI, false, onStreamError);

    expect(onStreamError).toHaveBeenCalledTimes(1);
    expect(onStreamError).toHaveBeenCalledWith(expect.objectContaining({ code: "protocol_incomplete" }), null, expect.any(Number));
  });

  it("passes OpenAI Responses events through and preserves terminal framing", async () => {
    const output = await runTransform([
      `event: response.created`,
      `data: ${JSON.stringify({ type: "response.created", response: { id: "resp_test", status: "in_progress" } })}`,
      "",
      `event: response.output_text.delta`,
      `data: ${JSON.stringify({ type: "response.output_text.delta", delta: "ok" })}`,
      "",
      `event: response.completed`,
      `data: ${JSON.stringify({ type: "response.completed", response: { id: "resp_test", status: "completed" } })}`,
      "",
      "data: [DONE]",
      "",
    ].join("\n"));

    expect(output).toContain("event: response.output_text.delta");
    expect(output).toContain("event: response.completed");
    expect(output.match(/data: \[DONE\]/g)).toHaveLength(1);
    expect(output).not.toContain("event: response.failed");
  });

  it("preserves event headers in Responses passthrough mode", async () => {
    const output = await runTransform([
      `event: response.created`,
      `data: ${JSON.stringify({ type: "response.created", response: { id: "resp_test", status: "in_progress" } })}`,
      "",
      `event: response.completed`,
      `data: ${JSON.stringify({ type: "response.completed", response: { id: "resp_test", status: "completed" } })}`,
      "",
    ].join("\n"), null, FORMATS.OPENAI_RESPONSES, FORMATS.OPENAI_RESPONSES, true);

    expect(output.match(/event: response.created/g)).toHaveLength(1);
    expect(output.match(/event: response.completed/g)).toHaveLength(1);
    expect(output.match(/data: \[DONE\]/g)).toHaveLength(1);
  });

  it("emits a response.failed event when a Responses stream closes before a terminal event", async () => {
    const output = await runTransform([
      `event: response.created`,
      `data: ${JSON.stringify({ type: "response.created", response: { id: "resp_test", status: "in_progress" } })}`,
      "",
      `event: response.output_text.delta`,
      `data: ${JSON.stringify({ type: "response.output_text.delta", delta: "partial" })}`,
      "",
    ].join("\n"), 7);

    expect(output).toContain("event: response.failed");
    expect(output).toContain('"type":"response.failed"');
    expect(output).not.toContain("data: null");
    expect(output).toContain("data: [DONE]");
  });

  it("does not add response.failed when a Responses stream already completed", async () => {
    const output = await runTransform([
      `event: response.completed`,
      `data: ${JSON.stringify({ type: "response.completed", response: { id: "resp_test", status: "completed" } })}`,
      "",
    ].join("\n"));

    expect(output).toContain("event: response.completed");
    expect(output).not.toContain("event: response.failed");
    expect(output).not.toContain("data: null");
    expect(output).toContain("data: [DONE]");
  });

  it("does not add response.failed when a Responses stream sends response.done", async () => {
    const output = await runTransform([
      `event: response.done`,
      `data: ${JSON.stringify({ type: "response.done", response: { id: "resp_test" } })}`,
      "",
    ].join("\n"));

    expect(output).toContain("event: response.done");
    expect(output).not.toContain("event: response.failed");
    expect(output).not.toContain("data: null");
    expect(output).toContain("data: [DONE]");
  });

  it("recognizes response.incomplete as a terminal failure", async () => {
    const output = await runTransform([
      `event: response.incomplete`,
      `data: ${JSON.stringify({ type: "response.incomplete", response: { id: "resp_test", incomplete_details: { reason: "max_output_tokens" } } })}`,
      "",
    ].join("\n"));

    expect(output).toContain("event: response.incomplete");
    expect(output).not.toContain("event: response.failed");
    expect(output).toContain("data: [DONE]");
  });

  it("emits response.failed before DONE when a Responses stream sends DONE without a terminal event", async () => {
    const output = await runTransform([
      `event: response.created`,
      `data: ${JSON.stringify({ type: "response.created", response: { id: "resp_test", status: "in_progress" } })}`,
      "",
      "data: [DONE]",
      "",
    ].join("\n"));

    expect(output.indexOf("event: response.failed")).toBeLessThan(output.indexOf("data: [DONE]"));
    expect(output.match(/data: \[DONE\]/g)).toHaveLength(1);
    expect(output).not.toContain("data: null");
  });

  it("emits only one DONE after duplicate terminal events", async () => {
    const output = await runTransform([
      `event: response.completed`,
      `data: ${JSON.stringify({ type: "response.completed", response: { id: "resp_test", status: "completed" } })}`,
      "",
      `event: response.done`,
      `data: ${JSON.stringify({ type: "response.done", response: { id: "resp_test", status: "completed" } })}`,
      "",
    ].join("\n"));

    expect(output.match(/data: \[DONE\]/g)).toHaveLength(1);
    expect(output.match(/"status":"completed"/g)).toHaveLength(2);
  });
});
