use std::borrow::Cow;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::{ToolSpec, task::SharedTaskManager, tool::Tool};

pub struct TaskCreateTool {
    manager: SharedTaskManager,
}

impl TaskCreateTool {
    pub fn new(manager: SharedTaskManager) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for TaskCreateTool {
    fn name(&self) -> Cow<'_, str> {
        "task_create".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task_create".to_string(),
            description: Some("Create a new persistent task.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "subject": { "type": "string" },
                    "description": { "type": "string" }
                },
                "required": ["subject"]
            }),
        }
    }

    async fn invoke(&mut self, input: &Value) -> Result<String> {
        let subject = input
            .get("subject")
            .and_then(Value::as_str)
            .context("Invalid subject")?
            .to_string();

        let description = input
            .get("description")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(ToOwned::to_owned);
        self.manager.create(subject, description)
    }
}
