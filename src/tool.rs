use std::{borrow::Cow, collections::HashMap, env, path::PathBuf, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::{
    ToolSpec,
    skill::SkillRegistry,
    tool::{
        bash::BashTool, edit_file::EditFileTool, load_skill::LoadSkillTool,
        read_file::ReadFileTool, sub_agent::SubAgentTool, todo::TodoManagerTool,
        write_file::WriteFileTool,
    },
};

pub mod bash;
pub mod edit_file;
pub mod load_skill;
pub mod read_file;
pub mod sub_agent;
pub mod todo;
pub mod write_file;

pub type Tools = HashMap<String, Box<dyn Tool>>;

#[async_trait]
pub trait Tool: Send + Sync {
    // name 必须和暴露给 LLM 的 tool_spec.name 保持一致。
    fn name(&self) -> Cow<'_, str>;
    // tool_spec 会随每次 LLM 请求发送，告诉模型该工具如何被调用。
    fn tool_spec(&self) -> ToolSpec;
    // invoke 接收模型生成的 JSON 参数，返回可写入 ToolResult 的纯文本结果。
    async fn invoke(&mut self, input: &Value) -> Result<String>;
}

pub fn agent_tools(registry: Arc<SkillRegistry>) -> Tools {
    HashMap::from([
        ("bash".into(), Box::new(BashTool) as Box<dyn Tool>),
        ("edit_file".into(), Box::new(EditFileTool) as Box<dyn Tool>),
        ("read_file".into(), Box::new(ReadFileTool) as Box<dyn Tool>),
        (
            "write_file".into(),
            Box::new(WriteFileTool) as Box<dyn Tool>,
        ),
        (
            "load_skill".into(),
            Box::new(LoadSkillTool::new(registry.clone())) as Box<dyn Tool>,
        ),
        (
            "task".into(),
            Box::new(SubAgentTool::new(registry.clone())) as Box<dyn Tool>,
        ),
        (
            "todo".into(),
            Box::new(TodoManagerTool::new()) as Box<dyn Tool>,
        ),
    ])
}

pub fn subagent_tools(registry: Arc<SkillRegistry>) -> Tools {
    HashMap::from([
        ("bash".into(), Box::new(BashTool) as Box<dyn Tool>),
        ("edit_file".into(), Box::new(EditFileTool) as Box<dyn Tool>),
        ("read_file".into(), Box::new(ReadFileTool) as Box<dyn Tool>),
        (
            "write_file".into(),
            Box::new(WriteFileTool) as Box<dyn Tool>,
        ),
        (
            "load_skill".into(),
            Box::new(LoadSkillTool::new(registry)) as Box<dyn Tool>,
        ),
    ])
}

fn safe_path(path: &str) -> Result<PathBuf> {
    // 文件工具只能访问当前工作区内的真实路径，避免模型读写工作区之外的文件。
    let cwd = env::current_dir()?;
    let full = cwd.join(path).canonicalize()?;
    if !full.starts_with(&cwd) {
        return Err(anyhow::anyhow!("Path escapes workspace"));
    }
    Ok(full)
}
