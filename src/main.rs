use std::{
    env,
    sync::{Arc, Mutex},
};

use anthropic_ai_sdk::types::message::{Message, Role};
use anyhow::Context;
use demo_cc::{
    LoopState, get_llm_client,
    hook::HookControl,
    invoke_hooks,
    memory::MemoryManager,
    permission::{PermissionManager, PermissionMode},
    skill::get_skill_registry,
    tool::agent_tools,
};
use inquire::{Select, Text};
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

    let mode = Select::new(
        "Permission mode:",
        vec![
            PermissionMode::Default,
            PermissionMode::Plan,
            PermissionMode::Auto,
        ],
    )
    .prompt()
    .context("An error happened or user cancelled the input.")?;

    let permission_manager = PermissionManager::try_new(mode)?;
    println!("[Permission mode: {}]", permission_manager.mode());

    let client = get_llm_client()?;

    let skills_dir = env::current_dir()?.join(SKILLS_DIR);
    let skill_registry = Arc::new(get_skill_registry(skills_dir)?);
    let memory_manager = Arc::new(Mutex::new(MemoryManager::init(
        env::current_dir()?.join(".memory"),
    )?));
    let tools = agent_tools(skill_registry.clone(), memory_manager.clone());
    let mut state = LoopState::new(
        client,
        tools,
        usize::MAX,
        permission_manager,
        skill_registry.clone(),
        memory_manager.clone(),
    );
    state.session_start(|_| {
        Box::pin(async {
            println!("--- Initializing...");
            Ok(HookControl::Continue)
        })
    });
    state.pre_tool(|_, tool_use| {
        println!("--- Before tool call: {tool_use:?}");
        Box::pin(async move { Ok(HookControl::Continue) })
    });
    state.post_tool(|_, tool_use, tool_result| {
        println!("--- After tool call: {tool_use:?}, result: {tool_result:?}");
        Box::pin(async move { Ok(HookControl::Continue) })
    });

    if let HookControl::Block(reason) = invoke_hooks!(SessionStart, &state)? {
        println!("--- Session blocked: {reason}");
        return Ok(());
    }

    loop {
        let prompt = Text::new("--- How can I help you? ---\n")
            .prompt()
            .context("An error happend or user cancelled the input.")?;
        let prompt = prompt.trim();

        if prompt.is_empty() {
            continue;
        }

        if ".exit".eq(prompt) {
            break;
        }

        if ".rules".eq(prompt) {
            for (index, rule) in state.permission_manager.rules().iter().enumerate() {
                println!("  {index}: {rule}")
            }
            continue;
        }

        if ".memory".eq(prompt) {
            let memory_manager = memory_manager
                .lock()
                .map_err(|_| anyhow::anyhow!("memory manager lock poisoned"))?;
            println!("{}", memory_manager.describe_memories());
            continue;
        }

        if prompt.trim().starts_with(".mode") {
            state
                .handle_mode_command(&prompt)
                .context("failed to switch permission mode")?;
            continue;
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
