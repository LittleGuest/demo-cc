use std::borrow::Cow;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::{ToolSpec, task::SharedTaskManager, tool::Tool};

pub struct TaskListTool {
    manager: SharedTaskManager,
}

impl TaskListTool {
    pub fn new(manager: SharedTaskManager) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for TaskListTool {
    fn name(&self) -> Cow<'_, str> {
        "task_list".into()
    }

    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task_list".to_string(),
            description: Some("List all tasks with status summary.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn invoke(&mut self, _: &Value) -> Result<String> {
        self.manager.list_all()
    }
}
