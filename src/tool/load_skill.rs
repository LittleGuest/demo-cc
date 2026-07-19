use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use tool_macros::tool;

use crate::tool::ToolContext;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoadSkillInput {
    #[schemars(description = "Name of the skill to load.")]
    pub name: String,
}

#[tool(
    name = "load_skill",
    description = "Load the full body of a named skill into the current context."
)]
pub async fn load_skill(ctx: ToolContext, input: LoadSkillInput) -> Result<String> {
    Ok(ctx.skill_registry.load_full_text(&input.name))
}
