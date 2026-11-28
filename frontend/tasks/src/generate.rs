//! Generation through the engine thread's queue.

use crate::api::{Chat, Finish, Generator, Params, Refused, Score, Sink, Step, Tokens, ToolChoice};
use crate::guided::Guides;
use crate::scopes;
use crate::{Error, Spool, Stops};
use std::sync::{Arc, Mutex, mpsc};
use tessera_bundle::tokenize::hf::Hf;
use tessera_bundle::{Message, RenderOptions, ResponseParser, Template};
use tessera_engine::{Event, Read, Reason, Request, Stacked};

/// A request for the engine thread, and its sink.
pub type Job = (Request, tessera_engine::Sink);

pub struct Generate {
    pub tokenizer: Hf,
    pub template: Option<Template>,
    pub response: Option<ResponseParser>,
    /// Ids that end an answer.
    pub stop: Vec<u32>,
    pub steer: Steer,
    /// Whether the engine samples. `false` refuses a temperature
    /// above zero.
    pub sampling: bool,
}

/// The guides an engine thread applies.
pub enum Steer {
    /// Penalties, and formats when guides are given.
    Guided(Option<Arc<dyn Guides>>),
    /// None: the named serving mode cannot apply guides.
    Unguided(&'static str),
}

impl Tokens for Generate {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }
}

impl Chat for Generate {
    fn render(&self, messages: &[Message], think: bool) -> Result<String, Error> {
        self.render_with(messages, think, None)
    }

    fn render_with(
        &self,
        messages: &[Message],
        think: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<String, Error> {
        let options = RenderOptions {
            think: (!think).then_some(false),
            reasoning_effort,
            ..RenderOptions::default()
        };
        self.render_options(messages, &options)
    }

    fn render_options(
        &self,
        messages: &[Message],
        options: &RenderOptions<'_>,
    ) -> Result<String, Error> {
        let template = self
            .template
            .as_ref()
            .ok_or("the model has no chat template")?;
        if messages.is_empty() {
            return Err("no messages".into());
        }
        if let Some(parser) = &self.response
            && !template.mentions("tool_calls")
        {
            return Ok(template.render_options(&parser.inline_calls(messages), options)?);
        }
        Ok(template.render_options(messages, options)?)
    }

    fn response(&self) -> Option<&ResponseParser> {
        self.response.as_ref()
    }
}

/// Answers through an engine thread's queue.
pub struct Queued {
    pub generate: Arc<Generate>,
    pub jobs: Mutex<mpsc::Sender<Job>>,
    /// Called after each job enters the queue.
    pub sent: Option<Box<dyn Fn() + Send + Sync>>,
}

impl Queued {
    fn send(&self, job: Job) -> Result<(), Error> {
        let jobs = self.jobs.lock().map_err(|_| "the engine stopped")?;
        jobs.send(job).map_err(|_| "the engine stopped")?;
        drop(jobs);
        if let Some(sent) = &self.sent {
            sent();
        }
        Ok(())
    }
}

impl Generator for Queued {
    fn supports_guides(&self) -> bool {
        matches!(self.generate.steer, Steer::Guided(_))
    }

    fn generate(&self, prompt: Vec<u32>, params: Params, mut sink: Sink) -> Result<(), Error> {
        let mut guides = Vec::new();
        let key = match params.penalties.neutral() {
            true => params.thinking.key().or(params.sequence.key()),
            false => Some("presence_penalty"),
        };
        let made = match &self.generate.steer {
            Steer::Guided(made) => made.as_ref(),
            Steer::Unguided(mode) => {
                match params
                    .format
                    .as_ref()
                    .map(|_| "response_format")
                    .or(key)
                    .or((params.min_tokens > 0).then_some("min_tokens"))
                    .or((params.tool_choice != ToolChoice::Auto).then_some("tool_choice"))
                {
                    Some(field) => {
                        let why = format!("{mode} does not support formats or penalties");
                        return Err(Box::new(Refused(why, field)));
                    }
                    None => None,
                }
            }
        };
        match (&params.format, made) {
            (None, _) => {}
            (Some(format), Some(made)) => guides.push(made.guide(format)?),
            (Some(_), None) => {
                let why = "this model was loaded without guided decoding";
                return Err(Box::new(Refused(why.into(), "response_format")));
            }
        }
        if !params.sampling.greedy() && !self.generate.sampling {
            let why = "this model was loaded for greedy decoding only";
            return Err(Box::new(Refused(why.into(), "temperature")));
        }
        if key.is_some() {
            let parser = self.generate.response.as_ref();
            guides.push(scopes::guide(
                parser,
                &self.generate.tokenizer,
                &params,
                &prompt,
            )?);
        }
        if let Some(guide) = scopes::tool_guide(
            self.generate.response.as_ref(),
            &self.generate.tokenizer,
            &params,
        )? {
            guides.push(guide);
        }
        let request = Request {
            tokens: prompt,
            max_tokens: params.max_tokens,
            min_tokens: params.min_tokens,
            stop: self.generate.stop.clone(),
            guide: Stacked::of(guides),
            sampling: params.sampling,
            ..Request::default()
        };
        let generate = self.generate.clone();
        let raw = params.raw;
        let (mut spool, mut stops) = (Spool::default(), Stops::new(params.stop));
        let (mut tokens, mut cached, mut ended) = (0, 0, false);
        // The engine sends `Done` after a sink refuses a token.
        let each = move |event: Event| {
            if ended {
                return false;
            }
            match event {
                Event::Kept(rows) => {
                    cached = rows;
                    true
                }
                Event::Scores(_) => true,
                Event::Token(id) => {
                    tokens += 1;
                    let piece = spool.push(id, |ids| match raw {
                        true => generate.tokenizer.decode_raw(ids),
                        false => generate.tokenizer.decode(ids),
                    });
                    if stops.is_buffering() {
                        let (parts, cut) = stops.push_tagged(tokens, &piece);
                        for (token, text) in parts {
                            if !sink(Step::Text { token, text }) {
                                ended = true;
                                return false;
                            }
                        }
                        if cut {
                            ended = true;
                            let _ = sink(Step::Done {
                                text: String::new(),
                                finish: Finish::Stop,
                                tokens,
                                cached,
                            });
                            return false;
                        }
                        true
                    } else {
                        let (text, cut) = stops.push(&piece);
                        let step = match cut {
                            false => Step::Token { id, text },
                            true => Step::Done {
                                text,
                                finish: Finish::Stop,
                                tokens,
                                cached,
                            },
                        };
                        ended = matches!(step, Step::Done { .. });
                        sink(step) && !ended
                    }
                }
                Event::Done(reason) => {
                    ended = true;
                    let finish = match reason {
                        Reason::Stop | Reason::Guide => Finish::Stop,
                        Reason::Length | Reason::Context => Finish::Length,
                    };
                    if stops.is_buffering() {
                        for (token, text) in stops.rest_tagged() {
                            if !sink(Step::Text { token, text }) {
                                return false;
                            }
                        }
                        let _ = sink(Step::Done {
                            text: String::new(),
                            finish,
                            tokens,
                            cached,
                        });
                    } else {
                        let _ = sink(Step::Done {
                            text: stops.rest(),
                            finish,
                            tokens,
                            cached,
                        });
                    }
                    false
                }
            }
        };
        self.send((request, Box::new(each)))
    }
}

impl Score for Queued {
    fn score(&self, prompt: Vec<u32>, reads: Vec<Read>) -> Result<Vec<Vec<f32>>, Error> {
        let request = Request {
            tokens: prompt,
            reads,
            ..Request::default()
        };
        let (sink, taken) = mpsc::channel();
        self.send((request, Box::new(move |e| sink.send(e).is_ok())))?;
        for event in taken {
            match event {
                Event::Scores(scores) => return Ok(scores),
                Event::Done(_) => break,
                _ => {}
            }
        }
        Err("the prompt and reads do not fit the context".into())
    }
}
