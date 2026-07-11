use std::{env, process::Stdio, time::Duration};

use anthropic_ai_sdk::{
    client::{AnthropicClient, AnthropicClientBuilder},
    types::message::{
        ContentBlock, CreateMessageParams, Message, MessageClient, MessageContent, MessageError,
        RequiredMessageParams, Role, StopReason, Tool,
    },
};
use anyhow::Context;
use inquire::Text;
use tokio::{process::Command, time::timeout};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

const SYSTEM_PROMPT: &str = r#"You are a coding agent.Use bash to inspect and change the workspace. Act first, then report clearly."#;

#[derive(Debug)]
struct LoopState {
    client: AnthropicClient,
    // 保存完整对话上下文，包括用户消息、模型回复和工具结果。
    pub context: Vec<Message>,
    // 记录当前 Agent 循环轮次，便于后续扩展调试或限制循环次数。
    turn_count: usize,
    // 记录状态转换原因，当前主要用于标记工具调用后的继续执行。
    transition_reason: Option<String>,
}

impl LoopState {
    fn new(client: AnthropicClient) -> Self {
        Self {
            client,
            context: Vec::new(),
            turn_count: 1,
            transition_reason: None,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化环境变量
    dotenvy::dotenv().ok();

    // 初始化tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let api_base_url = std::env::var("ANTHROPIC_BASE_URL").expect("ANTHROPIC_BASE_URL is not set");
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY is not set");

    // 创建客户端
    let client = AnthropicClientBuilder::new(api_key, "")
        .with_api_base_url(api_base_url)
        .build::<MessageError>()
        .context("can't create client")?;

    let mut state = LoopState::new(client);

    // 外层循环负责持续接收用户输入，每次输入都会触发一轮 Agent 执行。
    loop {
        let prompt = Text::new("--- How can I help you? ---\n").prompt()?;
        tracing::info!("{prompt}");
        if ".exit".eq(prompt.trim()) {
            break;
        }
        state.context.push(Message::new_text(Role::User, prompt));

        // 内层循环用于处理 LLM 发起的工具调用。
        // 只要 agent_loop 返回 true，就说明工具结果已回填，需要继续请求 LLM。
        while agent_loop(&mut state).await? {}

        let Some(final_content) = state.context.last() else {
            continue;
        };
        tracing::info!(
            "--- Final response: \n{}",
            extract_text(&final_content.content)
        );
    }

    Ok(())
}

fn extract_text(content: &MessageContent) -> String {
    // 最终输出只提取文本块，忽略工具调用等非文本内容。
    match content {
        MessageContent::Text { content } => content.clone(),
        MessageContent::Blocks { content } => content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

async fn agent_loop(state: &mut LoopState) -> anyhow::Result<bool> {
    // 每次请求都携带完整上下文、系统提示和可用工具定义。
    let request = CreateMessageParams::new(RequiredMessageParams {
        model: get_model()?,
        messages: state.context.clone(),
        max_tokens: 8000,
    })
    .with_system(SYSTEM_PROMPT)
    .with_tools(get_tools());

    let response = state.client.create_message(Some(&request)).await?;

    dbg!(&response.content);

    // 先把模型回复写入上下文，保证后续工具结果能和对应 tool_use 对齐。
    state.context.push(Message::new_blocks(
        Role::Assistant,
        response.content.clone(),
    ));

    // 如果模型没有要求调用工具，本轮 Agent 执行结束。
    if let Some(stop_reason) = response.stop_reason
        && !matches!(stop_reason, StopReason::ToolUse)
    {
        state.transition_reason = None;
        return Ok(false);
    }

    // 提取并执行工具调用；没有可执行工具时同样结束本轮。
    let Some(result) = execute_tool_call(&response.content).await else {
        state.transition_reason = None;
        return Ok(false);
    };

    // 工具结果以 User 消息形式回填给 LLM，驱动下一次 agent_loop。
    state.context.push(Message::new_blocks(Role::User, result));
    state.turn_count += 1;
    state.transition_reason = Some("tool_resule".into());
    Ok(true)
}

async fn execute_tool_call(content: &[ContentBlock]) -> Option<Vec<ContentBlock>> {
    let mut result = Vec::new();
    let mut has_tool_use = false;
    // 当前 Tool Runtime 只支持名为 bash 的工具。
    for block in content {
        if let ContentBlock::ToolUse { id, name, input } = block
            && name == "bash"
            && let Some(command) = input.get("command").and_then(|v| v.as_str())
        {
            has_tool_use = true;
            let output = run_bash(command).await;

            tracing::info!("Command {command} output: {output}");

            result.push(ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content: output,
            });
        }
    }

    if !has_tool_use {
        return None;
    }
    Some(result)
}

async fn run_bash(command: &str) -> String {
    // 简单拦截高风险命令，避免明显危险操作被执行。
    let dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
    if dangerous.iter().any(|item| command.contains(item)) {
        return "Error: Dangerous command blocked".into();
    }

    // 使用 sh -c 执行模型生成的命令，并捕获标准输出和标准错误。
    let child = match Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(err) => return format!("Error: {err}"),
    };

    let output_future = child.wait_with_output();
    // 为工具执行设置硬超时，避免 Agent 被长时间阻塞。
    match timeout(Duration::from_secs(120), output_future).await {
        Ok(Ok(output)) => {
            let combined = [output.stdout, output.stderr].concat();
            let out_str = String::from_utf8_lossy(&combined);
            let out_str = out_str.trim();
            if out_str.is_empty() {
                "(no output)".into()
            } else {
                out_str.chars().take(50000).collect()
            }
        }
        Ok(Err(err)) => format!("Error: {err}"),
        Err(_) => "Error: Timeout (120s)".into(),
    }
}

fn get_tools() -> Vec<Tool> {
    // 将本地 bash 能力暴露给 LLM，LLM 会按该 schema 生成工具调用参数。
    vec![Tool {
        name: "bash".into(),
        description: Some("Run a shell command in the current workspace.".into()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string"
                }
            },
            "required": ["command"]
        }),
    }]
}

fn get_model() -> anyhow::Result<String> {
    // 模型名称通过环境变量配置，避免写死在代码里。
    env::var("ANTHROPIC_MODEL").context("ANTHROPIC_MODEL is not set")
}
