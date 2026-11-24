//! The chat layout for `/v1/chat/completions`.

use tessera_bundle::{Message, RenderOptions, ResponseParser};
use tessera_tasks::Error;
use tessera_tasks::api::Chat;

/// Each message as `role:content`, joined.
pub(super) struct MockChat {
    pub(super) response: Option<ResponseParser>,
}

impl Chat for MockChat {
    fn render(&self, messages: &[Message], _think: bool) -> Result<String, Error> {
        let mut out = String::new();
        for m in messages {
            out.push_str(&m.role);
            out.push(':');
            if let Some(content) = &m.content {
                out.push_str(content);
            }
        }
        Ok(out)
    }

    fn render_options(
        &self,
        messages: &[Message],
        options: &RenderOptions<'_>,
    ) -> Result<String, Error> {
        let mut out = self.render(messages, options.think.unwrap_or(true))?;
        if !options.tools.is_empty() {
            out.push_str("|tools");
        }
        Ok(out)
    }

    fn response(&self) -> Option<&ResponseParser> {
        self.response.as_ref()
    }
}
