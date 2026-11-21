//! Little-endian words, written and read in the same order.

/// Bytes that words append to.
#[derive(Clone, Debug, Default)]
pub struct Writer(pub Vec<u8>);

/// Words from the front of `bytes`. None past the end.
#[derive(Clone, Copy, Debug)]
pub struct Reader<'a>(pub &'a [u8]);

/// A writer and reader method per type, single and counted.
macro_rules! words {
    ($($one:ident $many:ident),*) => {
        impl Writer {
            $(
                pub fn $one(&mut self, value: $one) -> &mut Self {
                    self.0.extend(value.to_le_bytes());
                    self
                }

                /// The count, then each value.
                pub fn $many(&mut self, values: &[$one]) -> &mut Self {
                    self.u32(values.len() as u32);
                    for v in values {
                        self.$one(*v);
                    }
                    self
                }
            )*
        }

        impl Reader<'_> {
            $(
                pub fn $one(&mut self) -> Option<$one> {
                    let (head, rest) = self.0.split_first_chunk()?;
                    self.0 = rest;
                    Some($one::from_le_bytes(*head))
                }

                pub fn $many(&mut self) -> Option<Vec<$one>> {
                    (0..self.u32()?).map(|_| self.$one()).collect()
                }
            )*
        }
    };
}

words!(u32 u32s, u64 u64s, f64 f64s);

impl Writer {
    /// Bytes as they are, with no length.
    pub fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.0.extend(bytes);
        self
    }

    /// The length, then the bytes.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.u32(bytes.len() as u32).raw(bytes)
    }
}

impl<'a> Reader<'a> {
    /// The next `n` bytes.
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    /// Bytes `Writer::bytes` wrote.
    pub fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()?;
        self.take(n as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
