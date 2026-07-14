use std::{collections::HashMap, env};

pub use anthropic_ai_sdk::types::message::Tool as ToolSpec;
use anthropic_ai_sdk::{
    client::{AnthropicClient, AnthropicClientBuilder},
    types::message::{
        ContentBlock, CreateMessageParams, Message, MessageClient, MessageContent, MessageError,
        RequiredMessageParams, Role, StopReason,
    },
};
use anyhow::Context as _;
use inquire::Select;

use crate::{
    compact::CompactState,
    hook::{
        Hook, HookControl, HookTypes, PostToolUseFn, PreToolUseFn, SessionStartFn, ToolResult,
        ToolUse,
    },
    permission::{PermissionBehavior, PermissionDecision, PermissionManager, PermissionMode},
    tool::Tools,
};

pub mod compact;
pub mod hook;
pub mod permission;
pub mod skill;
pub mod tool;

const PLAN_REMINDER_INTERVAL: usize = 3;
const CONTEXT_LIMIT: usize = 50000;

pub fn get_llm_client() -> anyhow::Result<AnthropicClient> {
    dotenvy::dotenv().ok();

    let api_base_url = std::env::var("ANTHROPIC_BASE_URL").expect("ANTHROPIC_BASE_URL is not set");
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY is not set");

    let client = AnthropicClientBuilder::new(api_key, "")
        .with_api_base_url(api_base_url)
        .build::<MessageError>()
        .context("can't create client")?;

    Ok(client)
}

pub fn get_model() -> anyhow::Result<String> {
    env::var("ANTHROPIC_MODEL").context("ANTHROPIC_MODEL is not set")
}

pub struct LoopState {
    client: AnthropicClient,
    // Anthropic Messages API 要求后续请求携带历史消息，这里保存完整会话轨迹。
    pub context: Vec<Message>,
    tools: Tools,
    system_prompt: String,
    max_round: usize,
    todo_rounds_since_update: usize,
    pub compact_state: CompactState,
    pub permission_manager: PermissionManager,
    pub hooks: Vec<Hook>,
}

impl LoopState {
    pub fn new(
        client: AnthropicClient,
        tools: Tools,
        system_prompt: impl Into<String>,
        max_round: usize,
        permission_manager: PermissionManager,
    ) -> Self {
        Self {
            client,
            context: Vec::new(),
            tools,
            system_prompt: system_prompt.into(),
            max_round,
            todo_rounds_since_update: 0,
            compact_state: CompactState::default(),
            permission_manager,
            hooks: Vec::new(),
        }
    }

    pub async fn agent_loop(&mut self) -> anyhow::Result<()> {
        for _ in 0..self.max_round {
            compact::micro_compact(&mut self.context);
            if compact::estimate_context_size(&self.context) > CONTEXT_LIMIT {
                println!("[auto compact]");
                self.compact_history(None).await?;
            }

            // 每次请求前先规范化历史消息，避免孤立 tool_use 或连续同角色消息破坏 API 约束。
            let request = CreateMessageParams::new(RequiredMessageParams {
                model: get_model()?,
                messages: self.normalize_messages(),
                max_tokens: 8000,
            })
            .with_system(&self.system_prompt)
            .with_tools(self.tools.values().map(|tool| tool.tool_spec()).collect());

            let response = self.client.create_message(Some(&request)).await?;

            self.context.push(Message::new_blocks(
                Role::Assistant,
                response.content.clone(),
            ));

            if let Some(stop_reason) = response.stop_reason
                && !matches!(stop_reason, StopReason::ToolUse)
            {
                return Ok(());
            }

            self.execute_tool_call(&response.content).await?;
        }
        Ok(())
    }

    async fn execute_tool_call(&mut self, content: &[ContentBlock]) -> anyhow::Result<()> {
        let mut result = Vec::new();
        let mut used_todo = false;
        let mut manual_compact = false;
        let mut compact_focus = None;
        for block in content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                let mut tool_use = ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                };
                if let HookControl::Block(reason) = invoke_hooks!(PreToolUse, self, &mut tool_use)?
                {
                    result.push(ContentBlock::ToolResult {
                        tool_use_id: tool_use.id.clone(),
                        content: format!("Tool blocked by PreToolUse hook: {reason}"),
                    });
                    continue;
                }
                // 权限检查
                let decision = self.permission_manager.check(name, input);
                let output;
                match decision {
                    PermissionDecision {
                        behavior: PermissionBehavior::Deny,
                        reason,
                    } => {
                        output = format!("Permission denied: {reason}");
                        println!("  [DENIED] {name}: {reason}");
                    }
                    PermissionDecision {
                        behavior: PermissionBehavior::Allow,
                        ..
                    } => {
                        output = self.execute(name, input).await;
                    }
                    PermissionDecision {
                        behavior: PermissionBehavior::Ask,
                        ..
                    } => {
                        if self.permission_manager.ask_user(name, input)? {
                            output = self.execute(name, input).await;
                        } else {
                            output = format!("Permission denied by user for : {name}");
                            println!("  [USER DENIED] {name}");
                        }
                    }
                }
                let mut tool_result = ToolResult {
                    tool_use_id: tool_use.id.clone(),
                    content: output,
                };
                if let HookControl::Block(reason) =
                    invoke_hooks!(PostToolUse, self, &tool_use, &mut tool_result)?
                {
                    tool_result.content = format!("Tool blocked by PostToolUse hook: {reason}");
                }

                result.push(ContentBlock::ToolResult {
                    tool_use_id: tool_use.id.clone(),
                    content: tool_result.content,
                });

                if name == "todo" {
                    used_todo = true;
                }

                if name == "read_file"
                    && let Some(path) = input.get("path").and_then(|v| v.as_str())
                {
                    self.remember_recent_file(path);
                }

                if name == "compact" {
                    println!("[manual compact");
                    manual_compact = true;
                    compact_focus = input.get("focus").and_then(|v| v.as_str());
                }
            }
        }
        if used_todo {
            self.todo_rounds_since_update = 0;
        } else {
            self.note_round_without_update();
            if let Some(reminder) = self.reminder() {
                result.insert(0, ContentBlock::Text { text: reminder });
            }
        }
        self.context.push(Message::new_blocks(Role::User, result));

        if manual_compact {
            self.compact_history(compact_focus)
                .await
                .context("manual compact failed")?;
        }
        Ok(())
    }

    async fn execute(&mut self, tool_name: &str, input: &serde_json::Value) -> String {
        let Some(tool) = self.tools.get_mut(tool_name) else {
            return format!("Unknown tool: {tool_name}");
        };
        match tool.invoke(input).await {
            Ok(output) => {
                tracing::info!(
                    "Command:{tool_name}\narg:{input}\noutput:\n{output}\n",
                    output = output.chars().take(200).collect::<String>(),
                );
                output
            }
            Err(e) => {
                tracing::error!("Error invoking tool {tool_name}: {e}");
                format!("Error invoking tool {tool_name}: {e}")
            }
        }
    }

    pub fn session_start(&mut self, hook: impl SessionStartFn + 'static) {
        self.hooks.push(Hook::SessionStart(Box::new(hook)));
    }

    pub fn pre_tool(&mut self, hook: impl PreToolUseFn + 'static) {
        self.hooks.push(Hook::PreToolUse(Box::new(hook)));
    }

    pub fn post_tool(&mut self, hook: impl PostToolUseFn + 'static) {
        self.hooks.push(Hook::PostToolUse(Box::new(hook)));
    }

    pub fn hooks_by_type(&self, hook_type: HookTypes) -> Vec<&Hook> {
        self.hooks
            .iter()
            .filter(|hook| hook_type == (*hook).into())
            .collect()
    }

    pub fn handle_mode_command(&mut self, query: &str) -> anyhow::Result<()> {
        let parts = query.split_whitespace().collect::<Vec<_>>();
        let mode = if parts.len() == 2 {
            parts[1].parse::<PermissionMode>().with_context(|| {
                format!(
                    "unknown mode: {}. Usage: /mode <default|plan|auto>",
                    parts[1]
                )
            })?
        } else {
            Select::new(
                "Permission mode:",
                vec![
                    PermissionMode::Default,
                    PermissionMode::Plan,
                    PermissionMode::Auto,
                ],
            )
            .prompt()
            .context("An error happened or user cancelled the input.")?
        };
        self.permission_manager.set_mode(mode);
        println!("[Switched to {}]", self.permission_manager.mode());
        Ok(())
    }

    fn note_round_without_update(&mut self) {
        self.todo_rounds_since_update += 1;
    }

    fn reminder(&mut self) -> Option<String> {
        if self.todo_rounds_since_update >= PLAN_REMINDER_INTERVAL {
            Some("<reminder>Refresh your current plan before continuing.</reminder>".into())
        } else {
            None
        }
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
        let mut messages = Vec::new();
        let mut index = 0;

        while index < self.context.len() {
            let msg = &self.context[index];
            let tool_use_ids = match &msg.content {
                MessageContent::Blocks { content } if matches!(msg.role, Role::Assistant) => {
                    content
                        .iter()
                        .filter_map(|block| {
                            if let ContentBlock::ToolUse { id, .. } = block {
                                Some(id.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                }
                _ => Vec::new(),
            };

            if tool_use_ids.is_empty() {
                if let Some(msg) = Self::strip_orphan_tool_results(msg.clone()) {
                    messages.push(msg);
                }
                index += 1;
                continue;
            }

            messages.push(msg.clone());

            let mut existing_results = HashMap::new();
            let mut remaining_user_blocks = Vec::new();
            let mut consumed_next_user = false;

            if let Some(next_msg) = self.context.get(index + 1)
                && matches!(next_msg.role, Role::User)
            {
                consumed_next_user = true;
                match &next_msg.content {
                    MessageContent::Blocks { content } => {
                        for block in content {
                            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                                existing_results.insert(tool_use_id.clone(), block.clone());
                            } else {
                                remaining_user_blocks.push(block.clone());
                            }
                        }
                    }
                    MessageContent::Text { content } => {
                        remaining_user_blocks.push(ContentBlock::Text {
                            text: content.clone(),
                        });
                    }
                }
            }

            // tool_result 必须紧跟在对应 tool_use 的下一条 user 消息里，不能追加到历史末尾。
            let mut result_blocks = tool_use_ids
                .into_iter()
                .map(|id| {
                    existing_results
                        .remove(&id)
                        .unwrap_or(ContentBlock::ToolResult {
                            tool_use_id: id,
                            content: "(cancelled)".into(),
                        })
                })
                .collect::<Vec<_>>();
            result_blocks.extend(remaining_user_blocks);
            messages.push(Message::new_blocks(Role::User, result_blocks));

            index += if consumed_next_user { 2 } else { 1 };
        }

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

    fn strip_orphan_tool_results(msg: Message) -> Option<Message> {
        if !matches!(msg.role, Role::User) {
            return Some(msg);
        }

        match msg.content {
            MessageContent::Blocks { content } => {
                let content = content
                    .into_iter()
                    .filter(|block| !matches!(block, ContentBlock::ToolResult { .. }))
                    .collect::<Vec<_>>();

                if content.is_empty() {
                    None
                } else {
                    Some(Message::new_blocks(Role::User, content))
                }
            }
            MessageContent::Text { .. } => Some(msg),
        }
    }
}
