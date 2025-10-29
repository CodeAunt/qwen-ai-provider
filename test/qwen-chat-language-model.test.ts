/* eslint-disable dot-notation */
import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import {
  convertReadableStreamToArray,
  createTestServer,
} from "@ai-sdk/provider-utils/test"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QwenChatLanguageModel } from "../src/qwen-chat-language-model"
import { createQwen } from "../src/qwen-provider"

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
]

const URL = "https://my.api.com/v1/chat/completions"

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const provider = createQwen({
  // baseURL: "https://my.api.com/v1/",
  // headers: {
  //   Authorization: `Bearer test-api-key`,
  // },
  // apiKey: "",
})

const model = provider("qwen-chat")

describe("config", () => {
  it("should extract base name from provider string", () => {
    const model = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "qwen.beta",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("qwen")
  })

  it("should handle provider without dot notation", () => {
    const model = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "qwen-plus",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("qwen-plus")
  })

  it("should return empty for empty provider", () => {
    const model = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("")
  })
})

describe("doGenerate", () => {
  const server = createTestServer({
    [URL]: {},
  })

  beforeEach(() => {
    server.calls.length = 0
  })

  function prepareJsonResponse({
    content = "",
    reasoning_content = "",
    tool_calls,
    function_call,
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
    finish_reason = "stop",
    id = "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
    created = 1711115037,
    model = "qwen-chat",
  }: {
    content?: string
    reasoning_content?: string
    tool_calls?: Array<{
      id: string
      type: "function"
      function: {
        name: string
        arguments: string
      }
    }>
    function_call?: {
      name: string
      arguments: string
    }
    usage?: {
      prompt_tokens?: number
      total_tokens?: number
      completion_tokens?: number
    }
    finish_reason?: string
    created?: number
    id?: string
    model?: string
  } = {}) {
    server.urls[URL].response = {
      type: "json-value",
      body: {
        id,
        object: "chat.completion",
        created,
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content,
              reasoning_content,
              tool_calls,
              function_call,
            },
            finish_reason,
          },
        ],
        usage,
        system_fingerprint: "fp_3bc1b5746c",
      },
    }
  }

  it("should pass user setting to requests", async () => {
    prepareJsonResponse({ content: "Hello, World!" })
    const modelWithUser = provider("qwen-chat", {
      user: "test-user-id",
    })
    await modelWithUser.doGenerate({
      prompt: TEST_PROMPT,
    })
    expect(await server.calls[0]?.requestBodyJson).toMatchObject({
      user: "test-user-id",
    })
  })

  it("should extract text response", async () => {
    prepareJsonResponse({ content: "Hello, World!" })

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(content[0]).toStrictEqual({
      type: "text",
      text: "Hello, World!",
    })
  })

  it("should extract reasoning content", async () => {
    prepareJsonResponse({
      content: "Hello, World!",
      reasoning_content: "This is the reasoning behind the response",
    })

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    const textParts = content.filter(p => p.type === "text")
    expect(textParts).toHaveLength(2)
    expect(textParts[0].text).toStrictEqual(
      "This is the reasoning behind the response",
    )
    expect(textParts[1].text).toStrictEqual("Hello, World!")
  })

  it("should extract usage", async () => {
    prepareJsonResponse({
      content: "",
      usage: { prompt_tokens: 20, total_tokens: 25, completion_tokens: 5 },
    })

    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(usage).toStrictEqual({
      inputTokens: 20,
      outputTokens: 5,
      totalTokens: 25,
    })
  })

  it("should send additional response information", async () => {
    prepareJsonResponse({
      id: "test-id",
      created: 123,
      model: "test-model",
    })

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(response).toStrictEqual({
      id: "test-id",
      timestamp: new Date(123 * 1000),
      modelId: "test-model",
    })
  })

  it("should support partial usage", async () => {
    prepareJsonResponse({
      content: "",
      usage: { prompt_tokens: 20, total_tokens: 20 },
    })

    const { usage } = await model.doGenerate({

      prompt: TEST_PROMPT,
    })

    expect(usage).toStrictEqual({
      inputTokens: 20,
      outputTokens: Number.NaN,
      totalTokens: 20,
    })
  })

  it("should extract finish reason", async () => {
    prepareJsonResponse({
      content: "",
      finish_reason: "stop",
    })

    const response = await model.doGenerate({

      prompt: TEST_PROMPT,
    })

    expect(response.finishReason).toStrictEqual("stop")
  })

  it("should support unknown finish reason", async () => {
    prepareJsonResponse({
      content: "",
      finish_reason: "eos",
    })

    const response = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(response.finishReason).toStrictEqual("unknown")
  })

  it("should expose the raw response headers", async () => {
    prepareJsonResponse({ content: "" })

    // server.responseHeaders = {
    //   "test-header": "test-value",
    // }

    const { response } = await model.doGenerate({

      prompt: TEST_PROMPT,
    })

    expect(response?.headers).toStrictEqual({
      // default headers:
      "content-length": "335",
      "content-type": "application/json",

      // // custom header
      // "test-header": "test-value",
    })
  })

  it("should pass the model and the messages", async () => {
    prepareJsonResponse({ content: "" })

    await model.doGenerate({

      prompt: TEST_PROMPT,
    })

    expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    })
  })

  it("should pass settings", async () => {
    prepareJsonResponse()

    await provider("qwen-chat", {
      user: "test-user-id",
    }).doGenerate({

      prompt: TEST_PROMPT,
    })

    expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
      user: "test-user-id",
    })
  })

  it("should include provider-specific options", async () => {
    prepareJsonResponse()

    await provider("qwen-chat").doGenerate({

      providerOptions: {
        "test-provider": {
          someCustomOption: "test-value",
        },
      },
      prompt: TEST_PROMPT,
    })

    expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareJsonResponse()

    await provider("qwen-chat").doGenerate({

      providerOptions: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
      prompt: TEST_PROMPT,
    })

    expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    })
  })

  it("should pass tools and toolChoice", async () => {
    prepareJsonResponse({ content: "" })

    await model.doGenerate({

      // mode: {
      //   type: "regular",
      // },
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      toolChoice: {
        type: "tool",
        toolName: "test-tool",
      },
      prompt: TEST_PROMPT,
    })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
      tools: [
        {
          type: "function",
          function: {
            name: "test-tool",
            parameters: {
              type: "object",
              properties: { value: { type: "string" } },
              required: ["value"],
              additionalProperties: false,
              $schema: "http://json-schema.org/draft-07/schema#",
            },
          },
        },
      ],
      toolChoice: {
        type: "function",
        function: { name: "test-tool" },
      },
    })
  })

  it("should pass headers", async () => {
    prepareJsonResponse({ content: "" })

    const provider = createQwen({
      baseURL: "https://my.api.com/v1/",
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider("qwen-chat").doGenerate({

      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    const requestHeaders = await server.calls[0]?.requestHeaders

    expect(requestHeaders).toStrictEqual({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should parse tool results", async () => {
    prepareJsonResponse({
      tool_calls: [
        {
          id: "call_O17Uplv4lJvD6DVdIvFFeRMw",
          type: "function",
          function: {
            name: "test-tool",
            arguments: "{\"value\":\"Spark\"}",
          },
        },
      ],
    })

    const result = await model.doGenerate({

      // mode: {
      //   type: "regular",
      // },
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      toolChoice: {
        type: "tool",
        toolName: "test-tool",
      },
      prompt: TEST_PROMPT,
    })

    expect(result.content).toStrictEqual([
      {
        // args: "{\"value\":\"Spark\"}",
        input: "{\"value\":\"Spark\"}",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        // toolCallType: "function",
        type: "tool-call",
        toolName: "test-tool",
      },
    ])
  })

  describe("response format", () => {
    it("should not send a response_format when response format is text", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "test-provider",
          url: () => URL,
          headers: () => ({}),
          supportsStructuredOutputs: false,
        },
      )

      await model.doGenerate({

        prompt: TEST_PROMPT,
        responseFormat: { type: "text" },
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
      })
    })

    it("should forward json response format as \"json_object\" without schema", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = provider("qwen-plus")

      await model.doGenerate({

        prompt: TEST_PROMPT,
        responseFormat: { type: "json" },
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
        response_format: { type: "json_object" },
      })
    })

    it("should forward json response format as \"json_object\" and omit schema when structuredOutputs are disabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "test-provider",
          url: () => URL,
          headers: () => ({}),
          supportsStructuredOutputs: false,
        },
      )

      const { warnings } = await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        response_format: { type: "json_object" },
      })

      expect(warnings).toEqual([
        {
          details:
            "JSON response format schema is only supported with structuredOutputs",
          setting: "responseFormat",
          type: "unsupported-setting",
        },
      ])
    })

    it("should forward json response format as \"json_object\" and include schema when structuredOutputs are enabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "test-provider",
          url: () => URL,
          headers: () => ({}),
          supportsStructuredOutputs: true,
        },
      )

      const { warnings } = await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "response",
            schema: {
              type: "object",
              properties: { value: { type: "string" } },
              required: ["value"],
              additionalProperties: false,
              $schema: "http://json-schema.org/draft-07/schema#",
            },
          },
        },
      })

      expect(warnings).toEqual([])
    })

    it("should use json_schema & strict in object-json mode when structuredOutputs are enabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "test-provider",
          url: () => URL,
          headers: () => ({}),
          supportsStructuredOutputs: true,
        },
      )

      await model.doGenerate({
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
        prompt: TEST_PROMPT,
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "response",
            schema: {
              type: "object",
              properties: { value: { type: "string" } },
              required: ["value"],
              additionalProperties: false,
              $schema: "http://json-schema.org/draft-07/schema#",
            },
          },
        },
      })
    })

    it("should set name & description in object-json mode when structuredOutputs are enabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "test-provider",
          url: () => URL,
          headers: () => ({}),
          supportsStructuredOutputs: true,
        },
      )

      await model.doGenerate({
        responseFormat: {
          type: "json",
          name: "test-name",
          description: "test description",
          schema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
        prompt: TEST_PROMPT,
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "test-name",
            description: "test description",
            schema: {
              type: "object",
              properties: { value: { type: "string" } },
              required: ["value"],
              additionalProperties: false,
              $schema: "http://json-schema.org/draft-07/schema#",
            },
          },
        },
      })
    })

    it("should allow for undefined schema in object-json mode when structuredOutputs are enabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "test-provider",
          url: () => URL,
          headers: () => ({}),
          supportsStructuredOutputs: true,
        },
      )

      await model.doGenerate({
        responseFormat: {
          type: "json",
          name: "test-name",
          description: "test description",
        },
        prompt: TEST_PROMPT,
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        response_format: {
          type: "json_object",
        },
      })
    })

    it("should set strict in object-tool mode when structuredOutputs are enabled", async () => {
      prepareJsonResponse({
        tool_calls: [
          {
            id: "call_O17Uplv4lJvD6DVdIvFFeRMw",
            type: "function",
            function: {
              name: "test-tool",
              arguments: "{\"value\":\"Spark\"}",
            },
          },
        ],
      })

      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "test-provider",
          url: () => URL,
          headers: () => ({}),
          supportsStructuredOutputs: true,
        },
      )

      const result = await model.doGenerate({
        tools: [{
          type: "function",
          name: "test-tool",
          description: "test description",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        }],
        prompt: TEST_PROMPT,
      })

      expect(await server.calls[0]?.requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        toolChoice: { type: "function", function: { name: "test-tool" } },
        tools: [
          {
            type: "function",
            function: {
              name: "test-tool",
              description: "test description",
              inputSchema: {
                type: "object",
                properties: { value: { type: "string" } },
                required: ["value"],
                additionalProperties: false,
                $schema: "http://json-schema.org/draft-07/schema#",
              },
            },
          },
        ],
      })

      expect(result.content).toStrictEqual([
        {
          args: "{\"value\":\"Spark\"}",
          toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
          toolCallType: "function",
          toolName: "test-tool",
        },
      ])
    })
  })

  it("should send request body", async () => {
    prepareJsonResponse({ content: "" })

    const { request } = await model.doGenerate({

      prompt: TEST_PROMPT,
    })

    expect(request).toStrictEqual({
      body: "{\"model\":\"qwen-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}",
    })
  })
})

describe("doStream", () => {
  const streamServer = createTestServer({
    [URL]: {},
  })

  beforeEach(() => {
    streamServer.calls.length = 0
  })

  function prepareStreamResponse({
    content,
    finish_reason = "stop",
  }: {
    content: string[]
    finish_reason?: string
  }) {
    streamServer.urls[URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
        + `"system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n`,
        ...content.map((text) => {
          return (
            `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
            + `"system_fingerprint":null,"choices":[{"index":1,"delta":{"content":"${text}"},"finish_reason":null}]}\n\n`
          )
        }),
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
        + `"system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"${finish_reason}"}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"${finish_reason}"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }
  }

  it("should stream text deltas", async () => {
    prepareStreamResponse({
      content: ["Hello", ", ", "World!"],
      finish_reason: "stop",
    })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    // note: space moved to last chunk bc of trimming
    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
        modelId: "qwen-chat",
        timestamp: new Date("2023-12-15T16:17:00.000Z"),
      },
      { type: "text-start", id: "content", delta: "" },
      { type: "text-delta", id: "content", delta: "Hello" },
      { type: "text-delta", id: "content", delta: ", " },
      { type: "text-end", id: "content", delta: "World!" },
      {
        type: "finish",
        finishReason: "stop",
        usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
      },
    ])
  })

  it("should stream reasoning content before text deltas", async () => {
    streamServer.urls[URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think"},"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"reasoning_content":" about this"},"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"content":"Here's"},"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"content":" my response"},"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],`
        + `"usage":{"prompt_tokens":18,"completion_tokens":439,"total_tokens":457}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({

      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
        modelId: "qwen-chat",
        timestamp: new Date("2024-03-25T09:06:38.000Z"),
      },
      {
        type: "text-start",
        id: "reasoning",
        delta: "Let me think",
      },
      {
        type: "text-delta",
        id: "reasoning",
        delta: " about this",
      },
      {
        type: "text-delta",
        id: "content",
        delta: "Here's",
      },
      {
        type: "text-end",
        id: "content",
        delta: " my response",
      },
      {
        type: "finish",
        finishReason: "stop",
        usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
      },
    ])
  })

  it("should stream tool deltas", async () => {
    const server = createTestServer({
      [URL]: {},
    })
    server.urls[URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
        + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"value"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Spark"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"le"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Day"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"}"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
        modelId: "qwen-chat",
        timestamp: new Date("2024-03-25T09:06:38.000Z"),
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "{\"",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "value",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "\":\"",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "Spark",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "le",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: " Day",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "\"}",
      },
      {
        type: "tool-call",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        args: "{\"value\":\"Sparkle Day\"}",
      },
      {
        type: "finish",
        finishReason: "tool-calls",
        usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
      },
    ])
  })

  it("should stream tool call deltas when tool call arguments are passed in the first chunk", async () => {
    const server = createTestServer({
      [URL]: {},
    })
    server.urls[URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
        + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":"{\\""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"va"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"lue"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Spark"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"le"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Day"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"}"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
        modelId: "qwen-chat",
        timestamp: new Date("2024-03-25T09:06:38.000Z"),
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "{\"",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "va",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "lue",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "\":\"",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "Spark",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "le",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: " Day",
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "\"}",
      },
      {
        type: "tool-call",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        args: "{\"value\":\"Sparkle Day\"}",
      },
      {
        type: "finish",
        finishReason: "tool-calls",
        usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
      },
    ])
  })

  it("should not duplicate tool calls when there is an additional empty chunk after the tool call has been completed", async () => {
    const server = createTestServer({
      [URL]: {},
    })
    server.urls[URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":226,"completion_tokens":0}}\n\n`,
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"id":"chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",`
        + `"type":"function","index":0,"function":{"name":"searchGoogle"}}]},"logprobs":null,"finish_reason":null}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":233,"completion_tokens":7}}\n\n`,
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
        + `"function":{"arguments":"{\\"query\\": \\""}}]},"logprobs":null,"finish_reason":null}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":241,"completion_tokens":15}}\n\n`,
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
        + `"function":{"arguments":"latest"}}]},"logprobs":null,"finish_reason":null}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":242,"completion_tokens":16}}\n\n`,
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
        + `"function":{"arguments":" news"}}]},"logprobs":null,"finish_reason":null}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":243,"completion_tokens":17}}\n\n`,
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
        + `"function":{"arguments":" on"}}]},"logprobs":null,"finish_reason":null}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":244,"completion_tokens":18}}\n\n`,
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
        + `"function":{"arguments":" ai\\"}"}}]},"logprobs":null,"finish_reason":null}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":245,"completion_tokens":19}}\n\n`,
        // empty arguments chunk after the tool call has already been finished:
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
        + `"function":{"arguments":""}}]},"logprobs":null,"finish_reason":"tool_calls","stop_reason":128008}],`
        + `"usage":{"prompt_tokens":226,"total_tokens":246,"completion_tokens":20}}\n\n`,
        `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
        + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[],`
        + `"usage":{"prompt_tokens":226,"total_tokens":246,"completion_tokens":20}}\n\n`,
        `data: [DONE]\n\n`,
      ],
    }

    const { stream } = await model.doStream({
      tools: [
        {
          type: "function",
          name: "searchGoogle",
          inputSchema: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chat-2267f7e2910a4254bac0650ba74cfc1c",
        modelId: "meta/llama-3.1-8b-instruct:fp8",
        timestamp: new Date("2024-12-02T17:57:21.000Z"),
      },
      {
        type: "text-delta",
        delta: "",
      },
      {
        type: "tool-call-delta",
        toolCallId: "chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",
        toolCallType: "function",
        toolName: "searchGoogle",
        argsTextDelta: "{\"query\": \"",
      },
      {
        type: "tool-call-delta",
        toolCallId: "chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",
        toolCallType: "function",
        toolName: "searchGoogle",
        argsTextDelta: "latest",
      },
      {
        type: "tool-call-delta",
        toolCallId: "chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",
        toolCallType: "function",
        toolName: "searchGoogle",
        argsTextDelta: " news",
      },
      {
        type: "tool-call-delta",
        toolCallId: "chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",
        toolCallType: "function",
        toolName: "searchGoogle",
        argsTextDelta: " on",
      },
      {
        type: "tool-call-delta",
        toolCallId: "chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",
        toolCallType: "function",
        toolName: "searchGoogle",
        argsTextDelta: " ai\"}",
      },
      {
        type: "tool-call",
        toolCallId: "chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",
        toolCallType: "function",
        toolName: "searchGoogle",
        args: "{\"query\": \"latest news on ai\"}",
      },
      {
        type: "finish",
        finishReason: "tool-calls",
        usage: { inputTokens: 226, outputTokens: 20, totalTokens: 246 },
      },
    ])
  })

  it("should stream tool call that is sent in one chunk", async () => {
    const server = createTestServer({
      [URL]: {},
    })
    server.urls[URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
        + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":"{\\"value\\":\\"Sparkle Day\\"}"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
        modelId: "qwen-chat",
        timestamp: new Date("2024-03-25T09:06:38.000Z"),
      },
      {
        type: "tool-call-delta",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        argsTextDelta: "{\"value\":\"Sparkle Day\"}",
      },
      {
        type: "tool-call",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolCallType: "function",
        toolName: "test-tool",
        args: "{\"value\":\"Sparkle Day\"}",
      },
      {
        type: "finish",
        finishReason: "tool-calls",
        usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
      },
    ])
  })

  it("should handle unparsable stream parts", async () => {
    const server = createTestServer({
      [URL]: {},
    })
    server.urls[URL].response = {
      type: "stream-chunks",
      chunks: [`data: {unparsable}\n\n`, "data: [DONE]\n\n"],
    }

    const { stream } = await model.doStream({

      prompt: TEST_PROMPT,
    })

    const elements = await convertReadableStreamToArray(stream)

    expect(elements.length).toBe(2)
    expect(elements[0].type).toBe("error")
    expect(elements[1]).toStrictEqual({
      finishReason: "error",
      type: "finish",
      usage: {
        outputTokens: Number.NaN,
        inputTokens: Number.NaN,
      },
    })
  })

  it("should expose the raw response headers", async () => {
    prepareStreamResponse({ content: [] })

    // streamServer.urls[URL].response!.headers = {
    //   "test-header": "test-value",
    // }

    const { response } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(response?.headers).toStrictEqual({
      // default headers:
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "connection": "keep-alive",

      // custom header
      // "test-header": "test-value",
    })
  })

  it("should pass the messages and the model", async () => {
    prepareStreamResponse({ content: [] })

    await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(await streamServer.calls[0]?.requestBodyJson).toStrictEqual({
      stream: true,
      stream_options: {
        include_usage: true,
      },
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    })
  })

  it("should pass headers", async () => {
    prepareStreamResponse({ content: [] })

    const provider = createQwen({
      baseURL: "https://my.api.com/v1",
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider("qwen-chat").doStream({

      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    const requestHeaders = await streamServer.calls[0]?.requestHeaders

    expect(requestHeaders).toStrictEqual({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    prepareStreamResponse({ content: [] })

    await provider("qwen-chat").doStream({

      providerOptions: {
        "test-provider": {
          someCustomOption: "test-value",
        },
      },
      prompt: TEST_PROMPT,
    })

    expect(await streamServer.calls[0]?.requestBodyJson).toStrictEqual({
      stream: true,
      stream_options: {
        include_usage: true,
      },
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareStreamResponse({ content: [] })

    await provider("qwen-chat").doStream({

      providerOptions: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
      prompt: TEST_PROMPT,
    })

    expect(await streamServer.calls[0]?.requestBodyJson).toStrictEqual({
      stream: true,
      stream_options: {
        include_usage: true,
      },
      model: "qwen-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    })
  })

  it("should send request body", async () => {
    prepareStreamResponse({ content: [] })

    const { request } = await model.doStream({

      prompt: TEST_PROMPT,
    })

    expect(request).toStrictEqual({
      body: "{\"model\":\"qwen-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true,\"stream_options\":{\"include_usage\":true}}",
    })
  })
})

describe("doStream simulated streaming", () => {
  const simulatedStreamServer = createTestServer({
    [URL]: {},
  })

  beforeEach(() => {
    simulatedStreamServer.calls.length = 0
  })

  function prepareJsonResponse({
    content = "",
    reasoning_content = "",
    tool_calls,
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
    finish_reason = "stop",
    id = "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
    created = 1711115037,
    model = "qwen-plus",
  }: {
    content?: string
    reasoning_content?: string
    tool_calls?: Array<{
      id: string
      type: "function"
      function: {
        name: string
        arguments: string
      }
    }>
    usage?: {
      prompt_tokens?: number
      total_tokens?: number
      completion_tokens?: number
    }
    finish_reason?: string
    created?: number
    id?: string
    model?: string
  } = {}) {
    simulatedStreamServer.urls[URL].response = {
      type: "json-value",
      body: {
        id,
        object: "chat.completion",
        created,
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content,
              tool_calls,
              reasoning_content,
            },
            finish_reason,
          },
        ],
        usage,
        system_fingerprint: "fp_3bc1b5746c",
      },
    }
  }

  it("should stream text delta", async () => {
    prepareJsonResponse({ content: "Hello, World!", model: "o1-preview" })

    const model = provider.chatModel("o1", {
      simulateStreaming: true,
    })

    const { stream } = await model.doStream({

      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        modelId: "o1-preview",
        timestamp: expect.any(Date),
      },
      { type: "text-delta", delta: "Hello, World!" },
      {
        type: "finish",
        finishReason: "stop",
        usage: { inputTokens: 4, outputTokens: 30, totalTokens: 34 },
        // logprobs: undefined,
        // providerOptions: undefined,
      },
    ])
  })

  it("should stream reasoning content before text delta in simulated streaming", async () => {
    prepareJsonResponse({
      content: "Hello, World!",
      reasoning_content: "This is the reasoning",
      model: "o1-preview",
    })

    const model = provider.chatModel("o1", {
      simulateStreaming: true,
    })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: "response-metadata",
        id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        modelId: "o1-preview",
        timestamp: expect.any(Date),
      },
      {
        type: "text-delta",
        id: "wED8qbCxJDYzl25v",
        delta: "This is the reasoning",
      },
      {
        type: "text-delta",
        id: "AYwSXUgl9r4ThohB",
        delta: "Hello, World!",
      },
      {
        type: "finish",
        finishReason: "stop",
        usage: { inputTokens: 4, outputTokens: 30, totalTokens: 34 },
        // logprobs: undefined,
        // providerOptions: undefined,
      },
    ])
  })

  it("should stream tool calls", async () => {
    prepareJsonResponse({
      model: "o1-preview",
      tool_calls: [
        {
          id: "call_O17Uplv4lJvD6DVdIvFFeRMw",
          type: "function",
          function: {
            name: "test-tool",
            arguments: "{\"value\":\"Sparkle Day\"}",
          },
        },
      ],
    })

    const model = provider.chatModel("o1", {
      simulateStreaming: true,
    })

    const { stream } = await model.doStream({
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      prompt: TEST_PROMPT,
    })

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        headers: {
          "content-length": "482",
          "content-type": "application/json",
        },
        type: "response-metadata",
        id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        modelId: "o1-preview",
        timestamp: expect.any(Date),
      },
      {
        type: "tool-call",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        // toolCallType: "function",
        toolName: "test-tool",
        input: "{\"value\":\"Sparkle Day\"}",
      },
      {
        type: "finish",
        finishReason: "stop",
        usage: { inputTokens: 4, outputTokens: 30, totalTokens: 34 },
        // logprobs: undefined,
        // providerOptions: undefined,
      },
    ])
  })
})

describe("metadata extraction", () => {
  const testMetadataExtractor = {
    extractMetadata: ({ parsedBody }: { parsedBody: unknown }) => {
      if (
        typeof parsedBody !== "object"
        || !parsedBody
        || !("test_field" in parsedBody)
      ) {
        return undefined
      }
      return {
        test: {
          value: parsedBody.test_field as string,
        },
      }
    },
    createStreamExtractor: () => {
      let accumulatedValue: string | undefined

      return {
        processChunk: (chunk: unknown) => {
          if (
            typeof chunk === "object"
            && chunk
            && "choices" in chunk
            && Array.isArray(chunk.choices)
            && chunk.choices[0]?.finish_reason === "stop"
            && "test_field" in chunk
          ) {
            accumulatedValue = chunk.test_field as string
          }
        },
        buildMetadata: () =>
          accumulatedValue
            ? {
                test: {
                  value: accumulatedValue,
                },
              }
            : undefined,
      }
    },
  }

  describe("non-streaming", () => {
    describe("metadata extraction", () => {
      const server = createTestServer({
        [URL]: {},
      })

      beforeEach(() => {
        server.calls.length = 0
        server.urls[URL].response = {
          type: "json-value",
          body: {
            id: "chatcmpl-123",
            object: "chat.completion",
            created: 1711115037,
            model: "qwen-plus",
            choices: [
              {
                index: 0,
                message: {
                  role: "assistant",
                  content: [{ type: "text", text: "Hello" }],
                },
                finish_reason: "stop",
              },
            ],
            test_field: "test_value",
          },
        }
      })

      it("should process metadata from complete response", async () => {
        const model = new QwenChatLanguageModel(
          "qwen-plus",
          {},
          {
            provider: "test-provider",
            url: () => URL,
            headers: () => ({}),
            metadataExtractor: testMetadataExtractor,
          },
        )

        const result = await model.doGenerate({
          prompt: TEST_PROMPT,
        })

        expect(result.providerMetadata).toEqual({
          test: {
            value: "test_value",
          },
        })

        const requestBody = await server.calls[0]?.requestBodyJson
        expect(requestBody).toStrictEqual({
          model: "qwen-plus",
          messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
        })
      })
    })
  })

  describe("streaming", () => {
    describe("metadata streaming", () => {
      const server = createTestServer({
        [URL]: {},
      })

      beforeEach(() => {
        server.calls.length = 0
        server.urls[URL].response = {
          type: "stream-chunks",
          chunks: [
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"test_field\":\"test_value\"}\n\n",
            "data: [DONE]\n\n",
          ],
        }
      })

      it("should process metadata from streaming response", async () => {
        const model = new QwenChatLanguageModel(
          "qwen-plus",
          {},
          {
            provider: "test-provider",
            url: () => URL,
            headers: () => ({}),
            metadataExtractor: testMetadataExtractor,
          },
        )

        const result = await model.doStream({
          prompt: TEST_PROMPT,
        })

        const parts = await convertReadableStreamToArray(result.stream)
        const finishPart = parts.find(part => part.type === "finish")

        expect(finishPart?.providerMetadata).toEqual({
          test: {
            value: "test_value",
          },
        })

        const requestBody = await server.calls[0]?.requestBodyJson
        expect(requestBody).toStrictEqual({
          model: "qwen-plus",
          messages: [{ role: "user", content: "Hello" }],
          stream: true,
          stream_options: {
            include_usage: true,
          },
        })
      })
    })
  })
})
