use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::{
    ToolSpec,
    tool::{Tool, safe_path},
};

pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> Cow<'_, str> {
        "edit_file".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".to_string(),
            description: Some("Replace exact text in file.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_text": { "type": "string" },
                    "new_text": { "type": "string" }
                },
                "required": ["path", "old_text", "new_text"]
            }),
        }
    }

    async fn invoke(&self, input: &Value) -> Result<String> {
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .context("Invalid path")?;
        // 编辑操作和读写操作一样，只允许作用在当前工作区内。
        let path = safe_path(path)?;

        let old_text = input
            .get("old_text")
            .and_then(|v| v.as_str())
            .context("Invalid old_text")?;

        let new_text = input
            .get("new_text")
            .and_then(|v| v.as_str())
            .context("Invalid new_text")?;

        let content = fs::read_to_string(&path)
            .await
            .map_err(|e| anyhow::anyhow!("Error: {}", e))?;

        if !content.contains(old_text) {
            return Err(anyhow::anyhow!(
                "Error: Text not found in {}",
                path.display()
            ));
        }

        // 只替换第一处匹配，避免一次工具调用意外修改多个位置。
        let updated = content.replacen(old_text, new_text, 1);

        fs::write(&path, updated)
            .await
            .map_err(|e| anyhow::anyhow!("Error: {}", e))?;

        Ok(format!("Edited {}", path.display()))
    }
}
