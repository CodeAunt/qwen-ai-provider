import type {
  LanguageModelV2Prompt,
  SharedV2ProviderMetadata,
} from "@ai-sdk/provider"
import type { QwenChatPrompt } from "./qwen-api-types"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils"

// JSDoc for helper function to extract Qwen metadata.
/**
 * Extracts Qwen-specific metadata from a message.
 *
 * @param message - An object that may contain providerMetadata
 * @param message.providerMetadata - Provider-specific metadata containing Qwen configuration
 * @returns The Qwen metadata object or an empty object if none exists
 */

function getQwenMetadata(message: {
  providerOptions?: SharedV2ProviderMetadata
}) {
  return message?.providerOptions?.qwen ?? {}
}

/**
 * Converts a generic language model prompt to Qwen chat messages.
 *
 * @param prompt The language model prompt to convert.
 * @returns An array of Qwen chat messages.
 */
export function convertToQwenChatMessages(
  prompt: LanguageModelV2Prompt,
): QwenChatPrompt {
  const messages: QwenChatPrompt = []
  // Iterate over each prompt message.
  for (const { role, content, ...message } of prompt) {
    const metadata = getQwenMetadata({ ...message })
    switch (role) {
      case "system": {
        // System messages are sent directly with metadata.
        messages.push({ role: "system", content, ...metadata })
        break
      }

      case "user": {
        if (content.length === 1 && content[0].type === "text") {
          // For a single text element, simplify the conversion.
          messages.push({
            role: "user",
            content: content[0].text,
            ...getQwenMetadata(content[0]),
          })
          break
        }
        // For multiple content parts, process each part.
        messages.push({
          role: "user",
          content: content.map((part) => {
            const partMetadata = getQwenMetadata(part)
            switch (part.type) {
              case "text": {
                // Plain text conversion.
                return { type: "text", text: part.text, ...partMetadata }
              }
              case "file": {
                // Convert files (including images) and encode if necessary.
                // Check if it's an image file
                if (part.mediaType?.startsWith("image/")) {
                  // Build the data URL from the file data
                  const url = typeof part.data === "string"
                    ? part.data // Already a data URL or regular URL
                    : part.data instanceof Uint8Array
                      ? `data:${part.mediaType};base64,${convertUint8ArrayToBase64(part.data)}`
                      : part.data.toString() // URL object

                  return {
                    type: "image_url",
                    image_url: {
                      url,
                    },
                    ...partMetadata,
                  }
                }
                // For non-image files, throw unsupported error
                throw new UnsupportedFunctionalityError({
                  functionality: "Non-image file content parts in user messages",
                })
              }
              default: {
                const _exhaustiveCheck: never = part
                throw new Error(`Unsupported part type: ${_exhaustiveCheck}`)
              }
            }
          }),
          ...metadata,
        })

        break
      }

      case "assistant": {
        // Build text response and accumulate function/tool calls.
        let text = ""
        const toolCalls: Array<{
          id: string
          type: "function"
          function: { name: string, arguments: string }
        }> = []

        for (const part of content) {
          const partMetadata = getQwenMetadata(part)
          switch (part.type) {
            case "text": {
              // Append each text part.
              text += part.text
              break
            }
            case "tool-call": {
              // Convert tool calls to function calls with serialized arguments.
              toolCalls.push({
                id: part.toolCallId,
                type: "function",
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.input),
                },
                ...partMetadata,
              })
              break
            }
            case "reasoning": {
              // Treat reasoning as text
              text += part.text
              break
            }
            case "file":
            case "tool-result": {
              // These types should not appear in assistant messages
              throw new UnsupportedFunctionalityError({
                functionality: `${part.type} content parts in assistant messages`,
              })
            }
            default: {
              // This branch should never occur.
              const _exhaustiveCheck: never = part
              throw new Error(`Unsupported part: ${_exhaustiveCheck}`)
            }
          }
        }

        messages.push({
          role: "assistant",
          content: text,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          ...metadata,
        })

        break
      }

      case "tool": {
        // Process tool responses by converting output to JSON string.
        for (const toolResponse of content) {
          const toolResponseMetadata = getQwenMetadata(toolResponse)
          messages.push({
            role: "tool",
            tool_call_id: toolResponse.toolCallId,
            content: JSON.stringify(toolResponse.output),
            ...toolResponseMetadata,
          })
        }
        break
      }

      default: {
        // Ensure all roles are handled.
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  return messages
}
