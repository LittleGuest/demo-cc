use std::{env, sync::Arc};

use anthropic_ai_sdk::types::message::{Message, Role};
use anyhow::Context;
use demo_cc::{LoopState, get_llm_client, skill::get_skill_registry, tool::agent_tools};
use inquire::Text;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

const SKILLS_DIR: &str = "skills";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // 工具执行过程使用 tracing 记录调试信息，最终回复仍直接打印到终端。
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let client = get_llm_client()?;

    let system_prompt = format!(
        "You are a coding agent at {}. Use the task tool to delegate exploration or subtasks.",
        env::current_dir()?.display()
    );

    let skills_dir = env::current_dir()?.join(SKILLS_DIR);
    let skill_registry = Arc::new(get_skill_registry(skills_dir)?);
    let tools = agent_tools(skill_registry);
    let mut state = LoopState::new(client, tools, system_prompt, usize::MAX);

    loop {
        let prompt = Text::new("--- How can I help you? ---\n")
            .prompt()
            .context("An error happend or user cancelled the input.")?;

        if prompt.trim().is_empty() {
            continue;
        }

        if ".exit".eq(prompt.trim()) {
            break;
        }

        state.context.push(Message::new_text(Role::User, prompt));

        state.agent_loop().await?;

        let Some(final_content) = state.context.last() else {
            continue;
        };
        println!(
            "################## Final response ##################: \n{}",
            LoopState::extract_text(&final_content.content)
        );
    }

    Ok(())
}
