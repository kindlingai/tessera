//! Hugging Face `config.json` keys that several model readers share.

use std::marker::PhantomData;

/// A key whose value is a `T`.
pub struct Key<T> {
    pub name: &'static str,
    value: PhantomData<fn() -> T>,
}

impl<T> Key<T> {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            value: PhantomData,
        }
    }
}

impl<T> Clone for Key<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Key<T> {}

/// The value of a key that holds more keys.
pub enum Object {}

/// A key by name, or a declared key.
pub trait Named<T> {
    fn name(&self) -> &str;
}

impl<T> Named<T> for &str {
    fn name(&self) -> &str {
        self
    }
}

impl<T> Named<T> for Key<T> {
    fn name(&self) -> &str {
        self.name
    }
}

/// Expands `$each` over the shared keys.
// Row: constant, key, value type.
macro_rules! hf_keys {
    ($each:ident) => {
        $each! {
            (TEXT_CONFIG, "text_config", Object),
            (VISION_CONFIG, "vision_config", Object),
            (ROPE_PARAMETERS, "rope_parameters", Object),
            (SLIDING_ATTENTION, "sliding_attention", Object),
            (FULL_ATTENTION, "full_attention", Object),
            (MODEL_TYPE, "model_type", String),
            (VOCAB, "vocab_size", u64),
            (HIDDEN, "hidden_size", u64),
            (LAYERS, "num_hidden_layers", u64),
            (HEADS, "num_attention_heads", u64),
            (HEAD_DIM, "head_dim", u64),
            (FFN, "intermediate_size", u64),
            (KV_HEADS, "num_key_value_heads", u64),
            (PER_LAYER_INPUT, "hidden_size_per_layer_input", u64),
            (TOP_K, "num_experts_per_tok", u64),
            (LINEAR_KEY_HEADS, "linear_num_key_heads", u64),
            (LINEAR_VALUE_HEADS, "linear_num_value_heads", u64),
            (LINEAR_KEY_DIM, "linear_key_head_dim", u64),
            (LINEAR_VALUE_DIM, "linear_value_head_dim", u64),
            (LINEAR_WINDOW, "linear_conv_kernel_dim", u64),
            (LAYER_TYPES, "layer_types", Vec<String>),
            (USE_SLIDING_WINDOW, "use_sliding_window", bool),
            (SLIDING_WINDOW, "sliding_window", u64),
            (ROPE_THETA, "rope_theta", f32),
            (ROPE_TYPE, "rope_type", String),
            (ROPE_SCALING, "rope_scaling", Object),
            (PARTIAL_ROTARY, "partial_rotary_factor", f64),
            (EPS, "rms_norm_eps", f32),
            (HIDDEN_ACT, "hidden_act", String),
            (HIDDEN_ACTIVATION, "hidden_activation", String),
            (OUTPUT_GATE_TYPE, "output_gate_type", String),
            (ATTENTION_BIAS, "attention_bias", bool),
            (MLP_BIAS, "mlp_bias", bool),
            (TIED, "tie_word_embeddings", bool),
        }
    };
}

macro_rules! define_keys {
    ($(($id:ident, $name:literal, $ty:ty)),* $(,)?) => {
        $(pub const $id: Key<$ty> = Key::new($name);)*
    };
}

hf_keys!(define_keys);
