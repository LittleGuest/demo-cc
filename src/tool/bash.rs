use std::{borrow::Cow, process::Stdio, time::Duration};

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::{process::Command, time::timeout};

use crate::{ToolSpec, tool::Tool};

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> Cow<'_, str> {
        "bash".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".to_string(),
            description: Some("Run a shell command in the current workspace.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn invoke(&self, input: &Value) -> Result<String> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .context("Invalid command")?;

        // 这里只做最基础的高风险命令拦截，真正的权限边界仍应由运行环境控制。
        let dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
        if dangerous.iter().any(|item| command.contains(item)) {
            return Err(anyhow::anyhow!("Error: Dangerous command blocked"));
        }

        // 通过 sh -c 执行命令，保留管道、重定向等 shell 语义。
        let child = match Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return Err(anyhow::anyhow!("Error: {}", e)),
        };

        // 工具调用需要有硬超时，否则一次长时间命令会阻塞整个 Agent 循环。
        let output_future = child.wait_with_output();
        match timeout(Duration::from_secs(120), output_future).await {
            Ok(Ok(output)) => {
                // LLM 需要同时看到 stdout 和 stderr，才能判断命令是否成功以及下一步怎么做。
                let combined = [output.stdout, output.stderr].concat();
                let out_str = String::from_utf8_lossy(&combined);
                let trimmed = out_str.trim();

                if trimmed.is_empty() {
                    Ok("(no output)".to_string())
                } else {
                    // 按字符截断，避免把过长输出全部塞回上下文。
                    Ok(trimmed.chars().take(50000).collect())
                }
            }
            Ok(Err(e)) => Err(anyhow::anyhow!("Error: {}", e)),
            Err(_) => Err(anyhow::anyhow!("Error: Timeout (120s)")),
        }
    }
}
