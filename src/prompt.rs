use std::{
    borrow::Cow,
    sync::{LazyLock, RwLock},
};

use anyhow::{Context, Result};
use derive_builder::Builder;
use tera::Tera;

pub static TERA: LazyLock<RwLock<Tera>> = LazyLock::new(|| RwLock::new(Tera::default()));

#[derive(Clone, Debug)]
enum TemplateRef {
    OneOff(Cow<'static, str>),
    Tera(Cow<'static, str>),
}

#[derive(Clone, Debug)]
pub struct Prompt {
    template_ref: TemplateRef,
    context: Option<tera::Context>,
}

impl Prompt {
    pub fn extend(other: &Tera) -> Result<()> {
        let mut tera = TERA.write().unwrap();
        tera.extend(other)?;
        Ok(())
    }

    pub fn from_compiled_template(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            template_ref: TemplateRef::Tera(name.into()),
            context: None,
        }
    }

    #[must_use]
    pub fn with_context(mut self, new_context: impl Into<tera::Context>) -> Self {
        let context = self.context.get_or_insert_with(tera::Context::default);
        context.extend(new_context.into());
        self
    }

    #[must_use]
    pub fn with_context_value(mut self, key: &str, value: impl Into<tera::Value>) -> Self {
        let context = self.context.get_or_insert_with(tera::Context::default);
        context.insert(key, &value.into());
        self
    }

    pub fn render(&self) -> Result<String> {
        if self.context.is_none()
            && let TemplateRef::OneOff(ref template) = self.template_ref
        {
            return Ok(template.to_string());
        }

        let context = self
            .context
            .as_ref()
            .map_or_else(|| Cow::Owned(tera::Context::default()), Cow::Borrowed);

        match &self.template_ref {
            TemplateRef::OneOff(template) => {
                tera::Tera::one_off(template.as_ref(), &context, false)
                    .context("Failed to render on-off template")
            }
            TemplateRef::Tera(template) => TERA
                .read()
                .unwrap()
                .render(template.as_ref(), &context)
                .context("Failed to render template"),
        }
    }
}

impl From<&'static str> for Prompt {
    fn from(value: &'static str) -> Self {
        Prompt {
            template_ref: TemplateRef::OneOff(value.into()),
            context: None,
        }
    }
}

impl From<String> for Prompt {
    fn from(value: String) -> Self {
        Prompt {
            template_ref: TemplateRef::OneOff(value.into()),
            context: None,
        }
    }
}

impl From<SystemPrompt> for Prompt {
    fn from(value: SystemPrompt) -> Self {
        let SystemPrompt {
            role,
            skills_available,
            memory,
            claude_md,
            dynamic_context,
            memory_guidance,
            guidelines,
            constraints,
            template,
            additional,
        } = value;

        template
            .with_context_value("role", role)
            .with_context_value("skills_available", skills_available)
            .with_context_value("memory", memory)
            .with_context_value("claude_md", claude_md)
            .with_context_value("dynamic_context", dynamic_context)
            .with_context_value("memory_guidance", memory_guidance)
            .with_context_value("guidelines", guidelines)
            .with_context_value("constraints", constraints)
            .with_context_value("additional", additional)
    }
}

#[derive(Clone, Debug, Builder)]
#[builder(setter(into, strip_option))]
pub struct SystemPrompt {
    #[builder(default)]
    role: Option<String>,

    #[builder(default)]
    skills_available: Option<String>,

    #[builder(default)]
    memory: Option<String>,

    #[builder(default)]
    claude_md: Option<String>,

    #[builder(default)]
    dynamic_context: Option<String>,

    #[builder(default)]
    memory_guidance: Option<String>,

    #[builder(default, setter(custom))]
    guidelines: Vec<String>,

    #[builder(default, setter(custom))]
    constraints: Vec<String>,

    #[builder(default)]
    additional: Option<String>,

    #[builder(default=default_prompt_template() )]
    template: Prompt,
}

fn default_prompt_template() -> Prompt {
    include_str!("system_prompt_template.md").into()
}

impl SystemPrompt {
    pub fn builder() -> SystemPromptBuilder {
        SystemPromptBuilder::default()
    }

    pub fn to_prompt(&self) -> Prompt {
        self.clone().into()
    }

    pub fn with_added_guideline(&mut self, guideline: impl AsRef<str>) -> &mut Self {
        self.guidelines.push(guideline.as_ref().to_string());
        self
    }

    pub fn with_guidelines<T: IntoIterator<Item = S>, S: AsRef<str>>(
        &mut self,
        guidelines: T,
    ) -> &mut Self {
        self.guidelines = guidelines
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        self
    }

    pub fn with_added_constraint(&mut self, constraint: impl AsRef<str>) -> &mut Self {
        self.constraints.push(constraint.as_ref().to_string());
        self
    }

    pub fn with_constraint<T: IntoIterator<Item = S>, S: AsRef<str>>(
        &mut self,
        constraints: T,
    ) -> &mut Self {
        self.constraints = constraints
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        self
    }

    pub fn with_role(&mut self, role: impl Into<String>) -> &mut Self {
        self.role = Some(role.into());
        self
    }

    pub fn with_skills_available(&mut self, skills_available: impl Into<String>) -> &mut Self {
        self.skills_available = Some(skills_available.into());
        self
    }

    pub fn with_memory(&mut self, memory: impl Into<String>) -> &mut Self {
        self.memory = Some(memory.into());
        self
    }

    pub fn with_claude_md(&mut self, claude_md: impl Into<String>) -> &mut Self {
        self.claude_md = Some(claude_md.into());
        self
    }

    pub fn with_dynamic_context(&mut self, dynamic_context: impl Into<String>) -> &mut Self {
        self.dynamic_context = Some(dynamic_context.into());
        self
    }

    pub fn with_memory_guidance(&mut self, guidance: impl Into<String>) -> &mut Self {
        self.memory_guidance = Some(guidance.into());
        self
    }

    pub fn with_additional(&mut self, additional: impl Into<String>) -> &mut Self {
        self.additional = Some(additional.into());
        self
    }

    pub fn with_template(&mut self, template: impl Into<Prompt>) -> &mut Self {
        self.template = template.into();
        self
    }
}

impl From<String> for SystemPrompt {
    fn from(value: String) -> Self {
        Self {
            role: None,
            skills_available: None,
            memory: None,
            claude_md: None,
            dynamic_context: None,
            memory_guidance: None,
            guidelines: Vec::new(),
            constraints: Vec::new(),
            additional: None,
            template: value.into(),
        }
    }
}

impl From<&'static str> for SystemPrompt {
    fn from(value: &'static str) -> Self {
        Self {
            role: None,
            skills_available: None,
            memory: None,
            claude_md: None,
            dynamic_context: None,
            memory_guidance: None,
            guidelines: Vec::new(),
            constraints: Vec::new(),
            additional: None,
            template: value.into(),
        }
    }
}

impl From<SystemPrompt> for SystemPromptBuilder {
    fn from(val: SystemPrompt) -> Self {
        SystemPromptBuilder {
            role: Some(val.role),
            skills_available: Some(val.skills_available),
            memory: Some(val.memory),
            claude_md: Some(val.claude_md),
            dynamic_context: Some(val.dynamic_context),
            memory_guidance: Some(val.memory_guidance),
            guidelines: Some(val.guidelines),
            constraints: Some(val.constraints),
            additional: Some(val.additional),
            template: Some(val.template),
        }
    }
}

impl From<Prompt> for SystemPrompt {
    fn from(prompt: Prompt) -> Self {
        SystemPrompt {
            role: None,
            skills_available: None,
            memory: None,
            claude_md: None,
            dynamic_context: None,
            memory_guidance: None,
            guidelines: Vec::new(),
            constraints: Vec::new(),
            additional: None,
            template: prompt,
        }
    }
}

impl Default for SystemPrompt {
    fn default() -> Self {
        SystemPrompt {
            role: None,
            skills_available: None,
            memory: None,
            claude_md: None,
            dynamic_context: None,
            memory_guidance: None,
            guidelines: Vec::new(),
            constraints: Vec::new(),
            additional: None,
            template: default_prompt_template(),
        }
    }
}

impl SystemPromptBuilder {
    pub fn add_guideline(&mut self, guideline: &str) -> &mut Self {
        self.guidelines
            .get_or_insert_with(Vec::new)
            .push(guideline.to_string());
        self
    }

    pub fn add_constraint(&mut self, constraint: &str) -> &mut Self {
        self.constraints
            .get_or_insert_with(Vec::new)
            .push(constraint.to_string());
        self
    }

    pub fn guidelines<T: IntoIterator<Item = S>, S: AsRef<str>>(
        &mut self,
        guidelines: T,
    ) -> &mut Self {
        self.guidelines = Some(
            guidelines
                .into_iter()
                .map(|s| s.as_ref().to_string())
                .collect(),
        );
        self
    }

    pub fn constraints<T: IntoIterator<Item = S>, S: AsRef<str>>(
        &mut self,
        constraints: T,
    ) -> &mut Self {
        self.constraints = Some(
            constraints
                .into_iter()
                .map(|s| s.as_ref().to_string())
                .collect(),
        );
        self
    }
}
