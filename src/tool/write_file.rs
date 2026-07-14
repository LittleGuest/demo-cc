use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::{
    ToolSpec,
    tool::{Tool, safe_path_for_write},
};

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> Cow<'_, str> {
        "write_file".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".to_string(),
            description: Some("Write content to file.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn invoke(&mut self, input: &Value) -> Result<String> {
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .context("Invalid path")?;
        // 写入目标必须解析到当前工作区内，不要求文件已存在。
        let path = safe_path_for_write(path)?;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .context("Invalid content")?;

        // 允许模型创建新文件；父目录不存在时自动补齐。
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.ok();
        }

        fs::write(&path, content)
            .await
            .map_err(|e| anyhow::anyhow!("Error: {}", e))?;

        Ok(format!(
            "Wrote {} bytes to {}",
            content.len(),
            path.display()
        ))
    }
}
