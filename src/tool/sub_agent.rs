use std::borrow::Cow;

use anthropic_ai_sdk::types::message::{Message, Role};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::{
    LoopState, ToolSpec, get_llm_client,
    tool::{Tool, subagent_tools},
};

pub struct SubAgentTool;

impl SubAgentTool {
    async fn sub_agent_loop(prompt: &str, description: Option<&str>) -> Result<String> {
        println!("> task - ({}): {}", description.unwrap_or_default(), prompt);
        let client = get_llm_client()?;
        let tools = subagent_tools();
        let system_prompt = format!(
            "You are a coding subagent at {}. Complete the given task, then summarize your findings.",
            std::env::current_dir()?.display()
        );
        let mut state = LoopState::new(client, tools, system_prompt, 30);
        state.context.push(Message::new_text(Role::User, prompt));
        state.agent_loop().await?;
        let summary = state
            .context
            .iter()
            .rev()
            .find(|message| matches!(message.role, Role::Assistant))
            .map(|message| LoopState::extract_text(&message.content))
            .filter(|text| !text.is_empty())
            .unwrap_or_else(|| "(no summary)".into());
        Ok(summary)
    }
}

#[async_trait]
impl Tool for SubAgentTool {
    fn name(&self) -> Cow<'_, str> {
        "task".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task".to_string(),
            description: Some("Spawn a subagent with fresh context. It shares the filesystem but not conversation history.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {"type": "string", "description": "Short description of the task"}
                },
                "required": ["prompt"]
            }),
        }
    }

    async fn invoke(&mut self, input: &Value) -> Result<String> {
        let prompt = input
            .get("prompt")
            .and_then(|v| v.as_str())
            .context("Invalid prompt")?;
        let description = input.get("description").and_then(|v| v.as_str());
        Self::sub_agent_loop(prompt, description).await
    }
}
