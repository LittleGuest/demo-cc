use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::{ToolSpec, backgroud::SharedBackgroundManager, tool::Tool};

pub struct RunBackgroundTool {
    manager: SharedBackgroundManager,
}

impl RunBackgroundTool {
    pub fn new(manager: SharedBackgroundManager) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for RunBackgroundTool {
    fn name(&self) -> Cow<'_, str> {
        "run_background".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_background".to_string(),
            description: Some(
                "Run a shell command in the background and return a task id immediately."
                    .to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        }
    }

    async fn invoke(&mut self, input: &Value) -> Result<String> {
        let command = input
            .get("command")
            .and_then(Value::as_str)
            .context("Invalid command")?
            .to_string();
        self.manager.run(command)
    }
}
