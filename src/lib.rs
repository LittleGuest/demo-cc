pub mod tool;

use std::{collections::HashSet, env};

pub use anthropic_ai_sdk::types::message::Tool as ToolSpec;
use anthropic_ai_sdk::{
    client::AnthropicClient,
    types::message::{
        ContentBlock, CreateMessageParams, Message, MessageClient, MessageContent,
        RequiredMessageParams, Role, StopReason,
    },
};
use anyhow::Context as _;

use crate::tool::{Tools, tools};

const SYSTEM_PROMPT: &str = r#"You are a coding agent.
Use bash to inspect and change the workspace. Act first, then report clearly.
"#;

pub struct LoopState {
    client: AnthropicClient,
    // Anthropic Messages API 要求后续请求携带历史消息，这里保存完整会话轨迹。
    pub context: Vec<Message>,
    tools: Tools,
}

impl LoopState {
    pub fn new(client: AnthropicClient) -> Self {
        Self {
            client,
            context: Vec::new(),
            tools: tools(),
        }
    }

    pub async fn agent_loop(&mut self) -> anyhow::Result<()> {
        loop {
            // 每次请求前先规范化历史消息，避免孤立 tool_use 或连续同角色消息破坏 API 约束。
            let request = CreateMessageParams::new(RequiredMessageParams {
                model: Self::get_model()?,
                messages: self.normalize_messages(),
                max_tokens: 8000,
            })
            .with_system(SYSTEM_PROMPT)
            .with_tools(self.tools.values().map(|tool| tool.tool_spec()).collect());

            let response = self.client.create_message(Some(&request)).await?;

            // Assistant 消息必须先入库；如果其中包含 tool_use，下一条 User 消息才能回填对应结果。
            self.context.push(Message::new_blocks(
                Role::Assistant,
                response.content.clone(),
            ));

            // 非 ToolUse 停止原因表示 LLM 已经给出最终回复，本轮对话结束。
            if let Some(stop_reason) = response.stop_reason
                && !matches!(stop_reason, StopReason::ToolUse)
            {
                return Ok(());
            }

            // ToolResult 会作为 User 消息回填，驱动 LLM 基于工具结果继续生成。
            let tool_result = self.execute_tool_call(&response.content).await;

            self.context
                .push(Message::new_blocks(Role::User, tool_result));
        }
    }

    async fn execute_tool_call(&mut self, content: &[ContentBlock]) -> Vec<ContentBlock> {
        let mut result = Vec::new();
        for block in content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                // tool_use_id 必须沿用 LLM 返回的 id，否则 API 无法匹配工具调用和工具结果。
                let output = self.execute(name, input).await;

                result.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: output,
                });
            }
        }
        result
    }

    async fn execute(&mut self, tool_name: &str, input: &serde_json::Value) -> String {
        let Some(tool) = self.tools.get_mut(tool_name) else {
            return format!("Unknown tool: {tool_name}");
        };
        match tool.invoke(input).await {
            Ok(output) => {
                tracing::info!("Command:{tool_name}\narg:{input}\noutput:\n{output}\n");
                output
            }
            Err(e) => {
                tracing::error!("Error invoking tool {tool_name}: {e}");
                format!("Error invoking tool {tool_name}: {e}")
            }
        }
    }

    fn get_model() -> anyhow::Result<String> {
        // 模型名属于运行时配置，便于在不同模型之间切换而不改代码。
        env::var("ANTHROPIC_MODEL").context("ANTHROPIC_MODEL is not set")
    }

    pub fn extract_text(content: &MessageContent) -> String {
        // 终端只展示自然语言回复；tool_use 等结构化块留在上下文中供下一轮请求使用。
        match content {
            MessageContent::Text { content } => content.clone(),
            MessageContent::Blocks { content } => content
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn normalize_messages(&self) -> Vec<Message> {
        let mut messages = self.context.to_vec();

        // Anthropic 要求每个 tool_use 都有对应 tool_result，先收集已经回填过的结果。
        let mut existing_results = HashSet::new();
        for msg in &messages {
            if let MessageContent::Blocks { content } = &msg.content {
                for block in content {
                    if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                        existing_results.insert(tool_use_id.clone());
                    }
                }
            }
        }

        // 如果历史里存在未回填的 tool_use，用 cancelled 结果补齐，避免后续请求被拒绝。
        let mut extra_messages = Vec::new();
        for msg in &messages {
            if matches!(msg.role, Role::User) {
                continue;
            }

            if let MessageContent::Blocks { content } = &msg.content {
                for block in content {
                    if let ContentBlock::ToolUse { id, .. } = block
                        && !existing_results.contains(id)
                    {
                        extra_messages.push(Message::new_blocks(
                            Role::User,
                            vec![ContentBlock::ToolResult {
                                tool_use_id: id.clone(),
                                content: "(cancelled)".into(),
                            }],
                        ));
                    }
                }
            }
        }
        messages.extend(extra_messages);

        // Messages API 更适合 user/assistant 交替出现；这里合并连续同角色消息，保持结构稳定。
        let mut merged: Vec<Message> = Vec::new();
        for msg in messages {
            if let Some(last) = merged.last_mut()
                && matches!(
                    (last.role, msg.role),
                    (Role::User, Role::User) | (Role::Assistant, Role::Assistant)
                )
            {
                match (&mut last.content, msg.content) {
                    (
                        MessageContent::Blocks { content: prev },
                        MessageContent::Blocks { content: curr },
                    ) => {
                        prev.extend(curr);
                    }
                    (
                        MessageContent::Text { content: prev },
                        MessageContent::Text { content: curr },
                    ) => {
                        prev.push('\n');
                        prev.push_str(&curr);
                    }
                    (
                        MessageContent::Text { content: prev },
                        MessageContent::Blocks { content: curr },
                    ) => {
                        let mut new_blocks = vec![ContentBlock::Text { text: prev.clone() }];
                        new_blocks.extend(curr);
                        last.content = MessageContent::Blocks {
                            content: new_blocks,
                        };
                    }
                    (
                        MessageContent::Blocks { content: prev },
                        MessageContent::Text { content: curr },
                    ) => {
                        prev.push(ContentBlock::Text { text: curr });
                    }
                }
                continue;
            }
            merged.push(msg);
        }
        merged
    }
}
