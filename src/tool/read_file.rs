use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::{
    ToolSpec,
    tool::{Tool, safe_path},
};

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> Cow<'_, str> {
        "read_file".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".to_string(),
            description: Some("Read file contents.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn invoke(&mut self, input: &Value) -> Result<String> {
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .context("Invalid path")?;
        // 所有文件读取都先经过工作区路径校验。
        let path = safe_path(path)?;

        let limit = input.get("limit").and_then(|v| v.as_u64());

        let content = fs::read_to_string(path)
            .await
            .map_err(|e| anyhow::anyhow!("Error: {}", e))?;

        let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();

        // limit 按行截断，适合让 LLM 先浏览大文件的开头部分。
        if let Some(limit) = limit
            && (limit as usize) < lines.len()
        {
            let remaining = lines.len() - limit as usize;
            lines.truncate(limit as usize);
            lines.push(format!("... ({} more lines)", remaining));
        }

        let result = lines.join("\n");

        // 再做一次字符级上限保护，避免大文件内容撑爆上下文。
        Ok(result.chars().take(50000).collect())
    }
}
