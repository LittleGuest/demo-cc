use std::{borrow::Cow, str::FromStr};

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::{
    ToolSpec,
    task::{SharedTaskManager, TaskStatus, TaskUpdate},
    tool::Tool,
};

pub struct TaskUpdateTool {
    manager: SharedTaskManager,
}

impl TaskUpdateTool {
    pub fn new(manager: SharedTaskManager) -> Self {
        Self { manager }
    }

    fn parse_task_id_list(input: &Value, field: &str) -> Result<Vec<u64>> {
        let Some(values) = input.get(field) else {
            return Ok(Vec::new());
        };

        let values = values
            .as_array()
            .with_context(|| format!("Invalid {field}"))?;

        values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .with_context(|| format!("Invalid integer in {field}"))
            })
            .collect()
    }
}

#[async_trait]
impl Tool for TaskUpdateTool {
    fn name(&self) -> Cow<'_, str> {
        "task_update".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task_update".to_string(),
            description: Some("Update a task's status, owner, or dependencies.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_id": { "type": "integer" },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "deleted"]
                    },
                    "owner": {
                        "type": "string",
                        "description": "Set when a teammate claims the task"
                    },
                    "addBlockedBy": {
                        "type": "array",
                        "items": { "type": "integer" }
                    },
                    "addBlocks": {
                        "type": "array",
                        "items": { "type": "integer" }
                    }
                },
                "required": ["task_id"]
            }),
        }
    }

    async fn invoke(&mut self, input: &Value) -> Result<String> {
        let task_id = input
            .get("task_id")
            .and_then(Value::as_u64)
            .context("Invalid task_id")?;

        let status = input
            .get("status")
            .and_then(Value::as_str)
            .map(TaskStatus::from_str)
            .transpose()
            .map_err(|_| {
                anyhow::anyhow!("Invalid status. Use pending, in_progress, completed, or deleted")
            })?;

        let owner = input
            .get("owner")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        let add_blocked_by = Self::parse_task_id_list(input, "addBlockedBy")?;
        let add_blocks = Self::parse_task_id_list(input, "addBlocks")?;

        self.manager.update(
            task_id,
            TaskUpdate {
                status,
                owner,
                add_blocked_by,
                add_blocks,
            },
        )
    }
}
