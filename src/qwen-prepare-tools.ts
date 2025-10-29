import type {
  LanguageModelV2CallWarning,
  LanguageModelV2FunctionTool,
  LanguageModelV2ProviderDefinedTool,
  LanguageModelV2ToolChoice,
} from "@ai-sdk/provider"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"

/**
 * Prepares the tool configuration for language model generation.
 * @param param0 Object containing mode details and structured output flag.
 * @returns An object with tools, tool choice and any warnings.
 */
export function prepareTools({
  tools,
  toolChoice,
}: {
  tools?: Array<LanguageModelV2FunctionTool | LanguageModelV2ProviderDefinedTool>
  toolChoice?: LanguageModelV2ToolChoice
}): {
  tools:
    | undefined
    | Array<{
      type: "function"
      function: {
        name: string
        description: string | undefined
        inputSchema: unknown
      }
    }>
  toolChoice:
    | { type: "function", function: { name: string } }
    | "auto"
    | "none"
    | "required"
    | undefined
  toolWarnings: LanguageModelV2CallWarning[]
} {
  // Normalize tools array by converting empty array to undefined.
  const toolWarnings: LanguageModelV2CallWarning[] = []

  if (!tools?.length) {
    return { tools: undefined, toolChoice: undefined, toolWarnings }
  }

  const qwenCompatTools: Array<{
    type: "function"
    function: {
      name: string
      description: string | undefined
      inputSchema: unknown
    }
  }> = []

  // Process each tool and format for compatibility.
  for (const tool of tools) {
    if (tool.type === "provider-defined") {
      // Warn if the tool is provider-defined.
      toolWarnings.push({ type: "unsupported-tool", tool })
    }
    else {
      qwenCompatTools.push({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema,
        },
      })
    }
  }

  if (toolChoice == null) {
    return { tools: qwenCompatTools, toolChoice: undefined, toolWarnings }
  }

  const type = toolChoice.type

  // Determine tool choice strategy.
  switch (type) {
    case "auto":
    case "none":
    case "required":
      return { tools: qwenCompatTools, toolChoice: type, toolWarnings }
    case "tool":
      return {
        tools: qwenCompatTools,
        toolChoice: {
          type: "function",
          function: {
            name: toolChoice.toolName,
          },
        },
        toolWarnings,
      }
    default: {
      // Exhaustive check to ensure all cases are handled.
      const _exhaustiveCheck: never = type
      throw new UnsupportedFunctionalityError({
        functionality: `Unsupported tool choice type: ${_exhaustiveCheck}`,
      })
    }
  }
}
