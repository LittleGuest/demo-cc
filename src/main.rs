use anthropic_ai_sdk::{
    client::AnthropicClientBuilder,
    types::message::{Message, MessageError, Role},
};
use anyhow::Context;
use demo_cc::LoopState;
use inquire::Text;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 允许通过 .env 提供 API 地址、密钥和模型名称，方便本地开发运行。
    dotenvy::dotenv().ok();

    // 工具执行过程使用 tracing 记录调试信息，最终回复仍直接打印到终端。
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let api_base_url = std::env::var("ANTHROPIC_BASE_URL").expect("ANTHROPIC_BASE_URL is not set");
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY is not set");

    // 使用环境变量创建 LLM 客户端，实际请求逻辑由 LoopState 负责。
    let client = AnthropicClientBuilder::new(api_key, "")
        .with_api_base_url(api_base_url)
        .build::<MessageError>()
        .context("can't create client")?;

    let mut state = LoopState::new(client);

    // CLI 入口只负责收集用户输入；对话上下文和工具调用循环都交给 Agent 状态机。
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
            "--- Final response: \n{}",
            LoopState::extract_text(&final_content.content)
        );
    }

    Ok(())
}
