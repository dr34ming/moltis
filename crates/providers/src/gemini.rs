use std::pin::Pin;

use {async_trait::async_trait, futures::StreamExt, secrecy::ExposeSecret, tokio_stream::Stream};

use tracing::{debug, trace, warn};

use crate::model::{
    ChatMessage, CompletionResponse, LlmProvider, StreamEvent, ToolCall, Usage, UserContent,
};

/// Information about a Gemini model returned from the API.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiModelInfo {
    /// Full resource name (e.g., "models/gemini-2.0-flash")
    pub name: String,
    /// Human-readable display name
    #[serde(default)]
    pub display_name: String,
    /// Maximum input tokens (context window)
    #[serde(default)]
    pub input_token_limit: u32,
    /// Maximum output tokens
    #[serde(default)]
    pub output_token_limit: u32,
    /// Supported generation methods (e.g., "generateContent", "streamGenerateContent")
    #[serde(default)]
    pub supported_generation_methods: Vec<String>,
}

impl GeminiModelInfo {
    /// Extract the model ID from the full resource name.
    /// E.g., "models/gemini-2.0-flash" -> "gemini-2.0-flash"
    pub fn model_id(&self) -> &str {
        self.name.strip_prefix("models/").unwrap_or(&self.name)
    }

    /// Check if this model supports text generation.
    pub fn supports_generation(&self) -> bool {
        self.supported_generation_methods
            .iter()
            .any(|m| m == "generateContent")
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListModelsResponse {
    models: Vec<GeminiModelInfo>,
    #[serde(default)]
    next_page_token: Option<String>,
}

/// List available Gemini models using an API key.
///
/// Returns models that support text generation, sorted by name.
pub async fn list_models(api_key: &str) -> anyhow::Result<Vec<GeminiModelInfo>> {
    list_models_with_base_url(api_key, "https://generativelanguage.googleapis.com").await
}

/// List available Gemini models with a custom base URL.
pub async fn list_models_with_base_url(
    api_key: &str,
    base_url: &str,
) -> anyhow::Result<Vec<GeminiModelInfo>> {
    let client = reqwest::Client::new();
    let mut all_models = Vec::new();
    let mut page_token: Option<String> = None;

    loop {
        let mut url = format!("{}/v1beta/models", base_url);
        if let Some(ref token) = page_token {
            url.push_str(&format!("?pageToken={}", token));
        }

        let resp = client
            .get(&url)
            .header("x-goog-api-key", api_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Failed to list Gemini models: HTTP {status}: {body}");
        }

        let list_resp: ListModelsResponse = resp.json().await?;
        all_models.extend(list_resp.models);

        match list_resp.next_page_token {
            Some(token) if !token.is_empty() => page_token = Some(token),
            _ => break,
        }
    }

    // Filter to models that support generation and sort by name
    let mut models: Vec<_> = all_models
        .into_iter()
        .filter(|m| m.supports_generation())
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

pub struct GeminiProvider {
    api_key: secrecy::Secret<String>,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl GeminiProvider {
    pub fn new(api_key: secrecy::Secret<String>, model: String, base_url: String) -> Self {
        Self {
            api_key,
            model,
            base_url,
            client: reqwest::Client::new(),
        }
    }

    /// List available models using this provider's API key.
    pub async fn list_available_models(&self) -> anyhow::Result<Vec<GeminiModelInfo>> {
        list_models_with_base_url(self.api_key.expose_secret(), &self.base_url).await
    }
}

/// Convert tool schemas from the generic format to Gemini's functionDeclarations format.
///
/// Input format (generic):
/// ```json
/// { "name": "...", "description": "...", "parameters": { "type": "object", ... } }
/// ```
///
/// Output format (Gemini):
/// ```json
/// { "functionDeclarations": [{ "name": "...", "description": "...", "parameters": { ... } }] }
/// ```
fn to_gemini_tools(tools: &[serde_json::Value]) -> serde_json::Value {
    let declarations: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            // Convert JSON Schema "type": "object" to Gemini's "type": "OBJECT"
            let params = convert_json_schema_types(&t["parameters"]);
            serde_json::json!({
                "name": t["name"],
                "description": t["description"],
                "parameters": params,
            })
        })
        .collect();

    serde_json::json!({ "functionDeclarations": declarations })
}

/// Convert JSON Schema types (lowercase) to Gemini types (uppercase).
/// Recursively handles nested objects and arrays.
fn convert_json_schema_types(schema: &serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Object(obj) => {
            let mut result = serde_json::Map::new();
            for (key, value) in obj {
                if key == "type" {
                    if let Some(type_str) = value.as_str() {
                        result.insert(
                            key.clone(),
                            serde_json::Value::String(type_str.to_uppercase()),
                        );
                    } else {
                        result.insert(key.clone(), value.clone());
                    }
                } else if key == "properties" {
                    // Properties is an object where each value is a schema that needs conversion
                    if let serde_json::Value::Object(props) = value {
                        let converted_props: serde_json::Map<String, serde_json::Value> = props
                            .iter()
                            .map(|(k, v)| (k.clone(), convert_json_schema_types(v)))
                            .collect();
                        result.insert(key.clone(), serde_json::Value::Object(converted_props));
                    } else {
                        result.insert(key.clone(), value.clone());
                    }
                } else if key == "items" {
                    // Items is a schema that needs conversion
                    result.insert(key.clone(), convert_json_schema_types(value));
                } else {
                    result.insert(key.clone(), value.clone());
                }
            }
            serde_json::Value::Object(result)
        },
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(convert_json_schema_types).collect())
        },
        _ => schema.clone(),
    }
}

/// Extract system instruction from messages, returning (system_text, remaining_messages).
fn extract_system_instruction(messages: &[ChatMessage]) -> (Option<String>, Vec<&ChatMessage>) {
    let mut system_text = None;
    let mut remaining = Vec::new();

    for msg in messages {
        if let ChatMessage::System { content } = msg {
            system_text = Some(content.clone());
        } else {
            remaining.push(msg);
        }
    }

    (system_text, remaining)
}

/// Convert messages to Gemini's content format.
///
/// Gemini uses:
/// - role: "user" for user messages and tool results
/// - role: "model" for assistant messages
/// - parts: array of { text: "..." } or { functionCall: {...} } or { functionResponse: {...} }
fn to_gemini_messages(messages: &[&ChatMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|msg| match msg {
            ChatMessage::System { .. } => {
                // System messages are handled separately via systemInstruction
                // This shouldn't be called with system messages, but handle gracefully
                serde_json::json!({
                    "role": "user",
                    "parts": [{ "text": "" }],
                })
            },
            ChatMessage::User { content } => {
                let parts = match content {
                    UserContent::Text(text) => {
                        vec![serde_json::json!({ "text": text })]
                    },
                    UserContent::Multimodal(parts) => parts
                        .iter()
                        .map(|p| match p {
                            crate::model::ContentPart::Text(text) => {
                                serde_json::json!({ "text": text })
                            },
                            crate::model::ContentPart::Image { media_type, data } => {
                                serde_json::json!({
                                    "inlineData": {
                                        "mimeType": media_type,
                                        "data": data,
                                    }
                                })
                            },
                        })
                        .collect(),
                };
                serde_json::json!({
                    "role": "user",
                    "parts": parts,
                })
            },
            ChatMessage::Assistant {
                content,
                tool_calls,
            } => {
                let mut parts = Vec::new();

                // Add text content if present
                if let Some(text) = content
                    && !text.is_empty()
                {
                    parts.push(serde_json::json!({ "text": text }));
                }

                // Add function calls
                for tc in tool_calls {
                    parts.push(serde_json::json!({
                        "functionCall": {
                            "name": &tc.name,
                            "args": &tc.arguments,
                        }
                    }));
                }

                // If no parts, add empty text to avoid empty parts array
                if parts.is_empty() {
                    parts.push(serde_json::json!({ "text": "" }));
                }

                serde_json::json!({
                    "role": "model",
                    "parts": parts,
                })
            },
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                // Try to parse content as JSON, fall back to wrapping as text
                let response: serde_json::Value = serde_json::from_str(content)
                    .unwrap_or_else(|_| serde_json::json!({ "result": content }));

                serde_json::json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_call_id,
                            "response": response,
                        }
                    }],
                })
            },
        })
        .collect()
}

/// Parse tool calls (functionCall) from Gemini response parts.
fn parse_tool_calls(parts: &[serde_json::Value]) -> Vec<ToolCall> {
    parts
        .iter()
        .filter_map(|part| {
            if let Some(fc) = part.get("functionCall") {
                let name = fc["name"].as_str().unwrap_or("").to_string();
                let args = fc["args"].clone();
                // Gemini doesn't use IDs for function calls, use the name as ID
                Some(ToolCall {
                    id: name.clone(),
                    name,
                    arguments: args,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Extract text content from Gemini response parts.
fn extract_text(parts: &[serde_json::Value]) -> Option<String> {
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect();

    if texts.is_empty() {
        None
    } else {
        Some(texts.join(""))
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    fn id(&self) -> &str {
        &self.model
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn context_window(&self) -> u32 {
        super::context_window_for_model(&self.model)
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> anyhow::Result<CompletionResponse> {
        // Extract system instruction and convert remaining messages
        let (system_text, conv_messages) = extract_system_instruction(messages);
        let gemini_messages = to_gemini_messages(&conv_messages);

        let mut body = serde_json::json!({
            "contents": gemini_messages,
            "generationConfig": {
                "maxOutputTokens": 8192,
            },
        });

        if let Some(ref sys) = system_text {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{ "text": sys }]
            });
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(vec![to_gemini_tools(tools)]);
        }

        debug!(
            model = %self.model,
            messages_count = gemini_messages.len(),
            tools_count = tools.len(),
            has_system = system_text.is_some(),
            "gemini complete request"
        );
        trace!(body = %serde_json::to_string(&body).unwrap_or_default(), "gemini request body");

        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url, self.model
        );

        let http_resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", self.api_key.expose_secret())
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = http_resp.status();
        if !status.is_success() {
            let body_text = http_resp.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "gemini API error");
            anyhow::bail!("Gemini API error HTTP {status}: {body_text}");
        }

        let resp = http_resp.json::<serde_json::Value>().await?;
        trace!(response = %resp, "gemini raw response");

        // Extract content from first candidate
        let parts = resp["candidates"][0]["content"]["parts"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let text = extract_text(&parts);
        let tool_calls = parse_tool_calls(&parts);

        let usage = Usage {
            input_tokens: resp["usageMetadata"]["promptTokenCount"]
                .as_u64()
                .unwrap_or(0) as u32,
            output_tokens: resp["usageMetadata"]["candidatesTokenCount"]
                .as_u64()
                .unwrap_or(0) as u32,
            ..Default::default()
        };

        Ok(CompletionResponse {
            text,
            tool_calls,
            usage,
        })
    }

    #[allow(clippy::collapsible_if)]
    fn stream(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Pin<Box<dyn Stream<Item = StreamEvent> + Send + '_>> {
        Box::pin(async_stream::stream! {
            // Extract system instruction and convert messages
            let (system_text, conv_messages) = extract_system_instruction(&messages);
            let gemini_messages = to_gemini_messages(&conv_messages);

            let mut body = serde_json::json!({
                "contents": gemini_messages,
                "generationConfig": {
                    "maxOutputTokens": 8192,
                },
            });

            if let Some(ref sys) = system_text {
                body["systemInstruction"] = serde_json::json!({
                    "parts": [{ "text": sys }]
                });
            }

            let url = format!(
                "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
                self.base_url, self.model
            );

            let resp = match self
                .client
                .post(&url)
                .header("x-goog-api-key", self.api_key.expose_secret())
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => {
                    if let Err(e) = r.error_for_status_ref() {
                        let status = e.status().map(|s| s.as_u16()).unwrap_or(0);
                        let body_text = r.text().await.unwrap_or_default();
                        yield StreamEvent::Error(format!("HTTP {status}: {body_text}"));
                        return;
                    }
                    r
                }
                Err(e) => {
                    yield StreamEvent::Error(e.to_string());
                    return;
                }
            };

            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut input_tokens: u32 = 0;
            let mut output_tokens: u32 = 0;

            while let Some(chunk) = byte_stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield StreamEvent::Error(e.to_string());
                        return;
                    }
                };
                buf.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events (data: {...}\n\n)
                while let Some(pos) = buf.find("\n\n") {
                    let block = buf[..pos].to_string();
                    buf = buf[pos + 2..].to_string();

                    for line in block.lines() {
                        let Some(data) = line.strip_prefix("data: ") else {
                            continue;
                        };

                        if let Ok(evt) = serde_json::from_str::<serde_json::Value>(data) {
                            // Extract usage metadata if present
                            if let Some(usage) = evt.get("usageMetadata") {
                                if let Some(pt) = usage["promptTokenCount"].as_u64() {
                                    input_tokens = pt as u32;
                                }
                                if let Some(ct) = usage["candidatesTokenCount"].as_u64() {
                                    output_tokens = ct as u32;
                                }
                            }

                            // Extract text delta from candidates
                            if let Some(parts) = evt["candidates"][0]["content"]["parts"].as_array() {
                                for part in parts {
                                    if let Some(text) = part["text"].as_str() {
                                        if !text.is_empty() {
                                            yield StreamEvent::Delta(text.to_string());
                                        }
                                    }
                                }
                            }

                            // Check for finish reason
                            if let Some(finish_reason) = evt["candidates"][0]["finishReason"].as_str() {
                                if finish_reason == "STOP" || finish_reason == "MAX_TOKENS" {
                                    yield StreamEvent::Done(Usage { input_tokens, output_tokens, ..Default::default() });
                                    return;
                                }
                            }
                        }
                    }
                }
            }

            // If we reach here without a STOP, still emit Done with what we have
            yield StreamEvent::Done(Usage { input_tokens, output_tokens, ..Default::default() });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_json_schema_types_converts_type_to_uppercase() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" },
                "items": {
                    "type": "array",
                    "items": { "type": "number" }
                }
            }
        });

        let converted = convert_json_schema_types(&schema);

        assert_eq!(converted["type"], "OBJECT");
        assert_eq!(converted["properties"]["name"]["type"], "STRING");
        assert_eq!(converted["properties"]["count"]["type"], "INTEGER");
        assert_eq!(converted["properties"]["items"]["type"], "ARRAY");
        assert_eq!(converted["properties"]["items"]["items"]["type"], "NUMBER");
    }

    #[test]
    fn to_gemini_tools_creates_function_declarations() {
        let tools = vec![serde_json::json!({
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }
        })];

        let gemini_tools = to_gemini_tools(&tools);

        assert!(gemini_tools["functionDeclarations"].is_array());
        let decls = gemini_tools["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], "get_weather");
        assert_eq!(decls[0]["description"], "Get weather for a location");
        assert_eq!(decls[0]["parameters"]["type"], "OBJECT");
    }

    #[test]
    fn extract_system_instruction_separates_system_message() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
        ];

        let (system, remaining) = extract_system_instruction(&messages);

        assert_eq!(system, Some("You are helpful".to_string()));
        assert_eq!(remaining.len(), 2);
        assert!(matches!(remaining[0], ChatMessage::User { .. }));
        assert!(matches!(remaining[1], ChatMessage::Assistant { .. }));
    }

    #[test]
    fn to_gemini_messages_converts_user_message() {
        let msg = ChatMessage::user("Hello");
        let messages = vec![&msg];

        let gemini = to_gemini_messages(&messages);

        assert_eq!(gemini.len(), 1);
        assert_eq!(gemini[0]["role"], "user");
        assert_eq!(gemini[0]["parts"][0]["text"], "Hello");
    }

    #[test]
    fn to_gemini_messages_converts_assistant_message() {
        let msg = ChatMessage::assistant("Hi there");
        let messages = vec![&msg];

        let gemini = to_gemini_messages(&messages);

        assert_eq!(gemini.len(), 1);
        assert_eq!(gemini[0]["role"], "model");
        assert_eq!(gemini[0]["parts"][0]["text"], "Hi there");
    }

    #[test]
    fn to_gemini_messages_converts_assistant_with_tool_calls() {
        let msg = ChatMessage::assistant_with_tools(None, vec![ToolCall {
            id: "call_123".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"location": "Boston"}),
        }]);
        let messages = vec![&msg];

        let gemini = to_gemini_messages(&messages);

        assert_eq!(gemini.len(), 1);
        assert_eq!(gemini[0]["role"], "model");
        let parts = gemini[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["functionCall"]["name"], "get_weather");
        assert_eq!(parts[0]["functionCall"]["args"]["location"], "Boston");
    }

    #[test]
    fn to_gemini_messages_converts_tool_result() {
        let msg = ChatMessage::tool("get_weather", r#"{"temperature": 72}"#);
        let messages = vec![&msg];

        let gemini = to_gemini_messages(&messages);

        assert_eq!(gemini.len(), 1);
        assert_eq!(gemini[0]["role"], "user");
        let parts = gemini[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionResponse"]["name"], "get_weather");
        assert_eq!(parts[0]["functionResponse"]["response"]["temperature"], 72);
    }

    #[test]
    fn parse_tool_calls_extracts_function_calls() {
        let parts = vec![
            serde_json::json!({ "text": "I'll check the weather" }),
            serde_json::json!({
                "functionCall": {
                    "name": "get_weather",
                    "args": { "location": "Boston" }
                }
            }),
        ];

        let tool_calls = parse_tool_calls(&parts);

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "get_weather");
        assert_eq!(tool_calls[0].arguments["location"], "Boston");
    }

    #[test]
    fn extract_text_combines_text_parts() {
        let parts = vec![
            serde_json::json!({ "text": "Hello " }),
            serde_json::json!({ "text": "world!" }),
        ];

        let text = extract_text(&parts);

        assert_eq!(text, Some("Hello world!".to_string()));
    }

    #[test]
    fn extract_text_returns_none_for_empty_parts() {
        let parts: Vec<serde_json::Value> = vec![];
        assert_eq!(extract_text(&parts), None);

        let parts = vec![serde_json::json!({ "functionCall": { "name": "test" } })];
        assert_eq!(extract_text(&parts), None);
    }

    #[test]
    fn provider_supports_tools() {
        let provider = GeminiProvider::new(
            secrecy::Secret::new("test-key".into()),
            "gemini-2.0-flash".into(),
            "https://example.com".into(),
        );
        assert!(provider.supports_tools());
    }

    #[test]
    fn provider_returns_correct_name_and_id() {
        let provider = GeminiProvider::new(
            secrecy::Secret::new("test-key".into()),
            "gemini-2.0-flash".into(),
            "https://example.com".into(),
        );
        assert_eq!(provider.name(), "gemini");
        assert_eq!(provider.id(), "gemini-2.0-flash");
    }

    #[test]
    fn provider_context_window_uses_lookup() {
        let provider = GeminiProvider::new(
            secrecy::Secret::new("test-key".into()),
            "gemini-2.0-flash".into(),
            "https://example.com".into(),
        );
        assert_eq!(provider.context_window(), 1_000_000);
    }

    // ── Model listing tests ──────────────────────────────────────────────────

    #[test]
    fn gemini_model_info_model_id_strips_prefix() {
        let info = GeminiModelInfo {
            name: "models/gemini-2.0-flash".into(),
            display_name: "Gemini 2.0 Flash".into(),
            input_token_limit: 1_000_000,
            output_token_limit: 8192,
            supported_generation_methods: vec!["generateContent".into()],
        };
        assert_eq!(info.model_id(), "gemini-2.0-flash");
    }

    #[test]
    fn gemini_model_info_model_id_handles_no_prefix() {
        let info = GeminiModelInfo {
            name: "gemini-2.0-flash".into(),
            display_name: "".into(),
            input_token_limit: 0,
            output_token_limit: 0,
            supported_generation_methods: vec![],
        };
        assert_eq!(info.model_id(), "gemini-2.0-flash");
    }

    #[test]
    fn gemini_model_info_supports_generation() {
        let info = GeminiModelInfo {
            name: "models/gemini-2.0-flash".into(),
            display_name: "".into(),
            input_token_limit: 0,
            output_token_limit: 0,
            supported_generation_methods: vec!["generateContent".into(), "embedContent".into()],
        };
        assert!(info.supports_generation());
    }

    #[test]
    fn gemini_model_info_not_generation_model() {
        let info = GeminiModelInfo {
            name: "models/text-embedding".into(),
            display_name: "".into(),
            input_token_limit: 0,
            output_token_limit: 0,
            supported_generation_methods: vec!["embedContent".into()],
        };
        assert!(!info.supports_generation());
    }

    #[test]
    fn list_models_response_deserializes() {
        let json = r#"{
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash",
                    "inputTokenLimit": 1000000,
                    "outputTokenLimit": 8192,
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]
                }
            ],
            "nextPageToken": "abc123"
        }"#;
        let resp: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 1);
        assert_eq!(resp.models[0].model_id(), "gemini-2.0-flash");
        assert_eq!(resp.models[0].display_name, "Gemini 2.0 Flash");
        assert_eq!(resp.models[0].input_token_limit, 1_000_000);
        assert!(resp.models[0].supports_generation());
        assert_eq!(resp.next_page_token, Some("abc123".to_string()));
    }

    #[test]
    fn list_models_response_handles_missing_fields() {
        let json = r#"{
            "models": [
                {
                    "name": "models/gemini-test"
                }
            ]
        }"#;
        let resp: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 1);
        assert_eq!(resp.models[0].model_id(), "gemini-test");
        assert_eq!(resp.models[0].display_name, "");
        assert_eq!(resp.models[0].input_token_limit, 0);
        assert_eq!(resp.models[0].output_token_limit, 0);
        assert!(!resp.models[0].supports_generation());
        assert!(resp.next_page_token.is_none());
    }

    #[test]
    fn list_models_response_handles_empty_next_page_token() {
        let json = r#"{
            "models": [],
            "nextPageToken": ""
        }"#;
        let resp: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert!(resp.models.is_empty());
        // Empty string should be treated as no token (our code checks for this)
        assert_eq!(resp.next_page_token, Some("".to_string()));
    }
}
