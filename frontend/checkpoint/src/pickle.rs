//! The pickle subset `torch.save` writes, read as data only.
//!
//! Globals come from a fixed list and no opcode runs code.

use crate::Error;
use std::collections::HashMap;

/// The most bytes a pickle may hold.
pub const MAX_PICKLE: usize = 64 << 20;
const MAX_DEPTH: u32 = 64;
/// The most value nodes memo reads may copy.
const MAX_COPIED: u64 = 1 << 24;
const MAX_RANK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DType {
    pub fn width(self) -> u64 {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::Bf16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// The type a `torch.<Name>Storage` global stands for.
    fn storage(name: &str) -> Option<DType> {
        Some(match name {
            "DoubleStorage" => DType::F64,
            "FloatStorage" => DType::F32,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::Bf16,
            "LongStorage" => DType::I64,
            "IntStorage" => DType::I32,
            "ShortStorage" => DType::I16,
            "CharStorage" => DType::I8,
            "ByteStorage" => DType::U8,
            "BoolStorage" => DType::Bool,
            _ => return None,
        })
    }
}

/// The only globals a pickle may name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Global {
    OrderedDict,
    Tensor,
    Parameter,
    Storage(DType),
}

fn global(module: &str, name: &str) -> Result<Global, Error> {
    let found = match (module, name) {
        ("collections", "OrderedDict") => Some(Global::OrderedDict),
        ("torch._utils", "_rebuild_tensor_v2") => Some(Global::Tensor),
        ("torch._utils", "_rebuild_parameter") => Some(Global::Parameter),
        ("torch", name) => DType::storage(name).map(Global::Storage),
        _ => None,
    };
    found.ok_or_else(|| refused(format_args!("global {module}.{name}")))
}

fn refused(what: impl std::fmt::Display) -> Error {
    Error::Format(format!("pickle: {what} is refused"))
}

fn bad(what: impl std::fmt::Display) -> Error {
    Error::Format(format!("pickle: {what}"))
}

/// A zip entry of `count` elements.
#[derive(Clone, Debug, PartialEq)]
pub struct Storage {
    pub key: String,
    pub dtype: DType,
    pub count: u64,
}

/// A strided view of a storage, in elements.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub storage: Storage,
    pub offset: u64,
    pub shape: Vec<usize>,
    pub stride: Vec<u64>,
}

impl View {
    /// One past the last element the view reads.
    pub fn end(&self) -> Option<u64> {
        let mut last = self.offset;
        for (&n, &s) in self.shape.iter().zip(&self.stride) {
            if n == 0 {
                return Some(self.offset);
            }
            last = last.checked_add((n as u64 - 1).checked_mul(s)?)?;
        }
        last.checked_add(1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(Global),
    Storage(Storage),
    Tensor(View),
}

/// A value with its nesting depth and node count.
#[derive(Clone)]
struct Slot {
    v: Value,
    depth: u32,
    size: u64,
}

impl Slot {
    fn leaf(v: Value) -> Self {
        Slot {
            v,
            depth: 0,
            size: 1,
        }
    }

    /// Takes `child`'s depth and size into this container.
    fn grow(&mut self, depth: u32, size: u64) -> Result<(), Error> {
        self.depth = self.depth.max(depth + 1);
        self.size = self.size.saturating_add(size);
        if self.depth > MAX_DEPTH {
            return Err(bad("values nest too deep"));
        }
        Ok(())
    }
}

fn tuple(items: Vec<Slot>) -> Result<Slot, Error> {
    let mut slot = Slot::leaf(Value::None);
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        slot.grow(item.depth, item.size)?;
        values.push(item.v);
    }
    slot.v = Value::Tuple(values);
    Ok(slot)
}

struct Machine<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<Slot>,
    marks: Vec<usize>,
    memo: HashMap<u32, Slot>,
    copied: u64,
}

impl<'a> Machine<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len());
        let end = end.ok_or_else(|| bad("ends early"))?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        Ok(self.take(N)?.try_into().expect("N bytes"))
    }

    fn line(&mut self) -> Result<&'a str, Error> {
        let rest = &self.bytes[self.at..];
        let n = rest.iter().position(|&b| b == b'\n');
        let line = self.take(n.ok_or_else(|| bad("a global has no newline"))?)?;
        self.at += 1;
        std::str::from_utf8(line).map_err(bad)
    }

    fn text(&mut self, n: usize) -> Result<Value, Error> {
        let s = std::str::from_utf8(self.take(n)?).map_err(bad)?;
        Ok(Value::Str(s.into()))
    }

    fn pop(&mut self) -> Result<Slot, Error> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() <= floor {
            return Err(bad("the stack runs empty"));
        }
        Ok(self.stack.pop().expect("checked"))
    }

    fn top(&mut self) -> Result<&mut Slot, Error> {
        self.stack
            .last_mut()
            .ok_or_else(|| bad("the stack runs empty"))
    }

    fn to_mark(&mut self) -> Result<Vec<Slot>, Error> {
        let mark = self.marks.pop().ok_or_else(|| bad("no mark"))?;
        Ok(self.stack.split_off(mark))
    }

    fn push(&mut self, v: Value) {
        self.stack.push(Slot::leaf(v));
    }

    fn put(&mut self, key: u32) -> Result<(), Error> {
        let slot = self.top()?.clone();
        self.memo.insert(key, slot);
        Ok(())
    }

    fn get(&mut self, key: u32) -> Result<(), Error> {
        let slot = self
            .memo
            .get(&key)
            .ok_or_else(|| bad("a memo key is unset"))?;
        self.copied = self.copied.saturating_add(slot.size);
        if self.copied > MAX_COPIED {
            return Err(bad("memo reads copy too much"));
        }
        self.stack.push(slot.clone());
        Ok(())
    }

    /// A memo put for `q` and `r`, a get for `h` and `j`.
    fn memo_op(&mut self, op: u8, key: u32) -> Result<(), Error> {
        match op {
            b'q' | b'r' => self.put(key),
            _ => self.get(key),
        }
    }

    /// Adds `items` to the list on top.
    fn append(&mut self, items: Vec<Slot>) -> Result<(), Error> {
        let list = self.top()?;
        for item in items {
            list.grow(item.depth, item.size)?;
            let Value::List(values) = &mut list.v else {
                return Err(refused("appending to a non-list"));
            };
            values.push(item.v);
        }
        Ok(())
    }

    /// Adds key, value pairs to the dict on top.
    fn set(&mut self, items: Vec<Slot>) -> Result<(), Error> {
        if items.len() % 2 != 0 {
            return Err(bad("a key has no value"));
        }
        let dict = self.top()?;
        let mut items = items.into_iter();
        while let (Some(k), Some(v)) = (items.next(), items.next()) {
            dict.grow(k.depth.max(v.depth), k.size.saturating_add(v.size))?;
            let Value::Dict(pairs) = &mut dict.v else {
                return Err(refused("setting items on a non-dict"));
            };
            pairs.push((k.v, v.v));
        }
        Ok(())
    }

    fn reduce(&mut self) -> Result<(), Error> {
        let args = self.pop()?;
        let call = self.pop()?;
        let (Value::Global(g), Value::Tuple(args)) = (call.v, args.v) else {
            return Err(refused("a call on a non-global"));
        };
        let out = match (g, args.as_slice()) {
            (Global::OrderedDict, []) => Value::Dict(Vec::new()),
            (Global::Tensor, [s, offset, shape, stride, _, _, ..]) if args.len() <= 7 => {
                Value::Tensor(rebuild(s, offset, shape, stride)?)
            }
            (Global::Parameter, [t @ Value::Tensor(_), _, _]) => t.clone(),
            (g, _) => return Err(refused(format_args!("{g:?} with these arguments"))),
        };
        self.push(out);
        Ok(())
    }

    fn persistent(&mut self) -> Result<(), Error> {
        let Value::Tuple(pid) = self.pop()?.v else {
            return Err(bad("a persistent id is not a tuple"));
        };
        let [
            Value::Str(tag),
            Value::Global(Global::Storage(dtype)),
            Value::Str(key),
            _,
            count,
        ] = pid.as_slice()
        else {
            return Err(refused("this persistent id"));
        };
        let plain = key.bytes().all(|b| b.is_ascii_alphanumeric());
        if tag != "storage" || key.is_empty() || !plain {
            return Err(refused(format_args!("storage key {key:?}")));
        }
        self.push(Value::Storage(Storage {
            key: key.clone(),
            dtype: *dtype,
            count: count_of(count)?,
        }));
        Ok(())
    }

    fn step(&mut self, op: u8) -> Result<Option<Value>, Error> {
        match op {
            0x80 => {
                let [proto] = self.fixed()?;
                if !(2..=4).contains(&proto) {
                    return Err(refused(format_args!("protocol {proto}")));
                }
            }
            0x95 => {
                let len = u64::from_le_bytes(self.fixed()?);
                if len > (self.bytes.len() - self.at) as u64 {
                    return Err(bad("a frame runs past the end"));
                }
            }
            b'.' => {
                let top = self.pop()?;
                if !self.stack.is_empty() || !self.marks.is_empty() {
                    return Err(bad("values remain at the end"));
                }
                return Ok(Some(top.v));
            }
            b'(' => {
                if self.marks.len() as u32 >= MAX_DEPTH {
                    return Err(bad("marks nest too deep"));
                }
                self.marks.push(self.stack.len());
            }
            b'}' => self.push(Value::Dict(Vec::new())),
            b']' => self.push(Value::List(Vec::new())),
            b')' => self.push(Value::Tuple(Vec::new())),
            b'N' => self.push(Value::None),
            0x88 => self.push(Value::Bool(true)),
            0x89 => self.push(Value::Bool(false)),
            b't' => {
                let items = self.to_mark()?;
                self.stack.push(tuple(items)?);
            }
            0x85..=0x87 => {
                let mut items = Vec::new();
                for _ in 0..op - 0x84 {
                    items.push(self.pop()?);
                }
                items.reverse();
                self.stack.push(tuple(items)?);
            }
            b'q' | b'h' => {
                let key = self.fixed::<1>()?[0] as u32;
                self.memo_op(op, key)?;
            }
            b'r' | b'j' => {
                let key = u32::from_le_bytes(self.fixed()?);
                self.memo_op(op, key)?;
            }
            0x94 => self.put(self.memo.len() as u32)?,
            b'K' => {
                let n = self.fixed::<1>()?[0] as i64;
                self.push(Value::Int(n));
            }
            b'M' => {
                let n = u16::from_le_bytes(self.fixed()?) as i64;
                self.push(Value::Int(n));
            }
            b'J' => {
                let n = i32::from_le_bytes(self.fixed()?) as i64;
                self.push(Value::Int(n));
            }
            0x8a => {
                let [n] = self.fixed()?;
                let int = long(self.take(n as usize)?)?;
                self.push(Value::Int(int));
            }
            b'G' => {
                let x = f64::from_be_bytes(self.fixed()?);
                self.push(Value::Float(x));
            }
            b'X' => {
                let n = u32::from_le_bytes(self.fixed()?) as usize;
                let v = self.text(n)?;
                self.push(v);
            }
            0x8c => {
                let [n] = self.fixed()?;
                let v = self.text(n as usize)?;
                self.push(v);
            }
            b'a' => {
                let item = self.pop()?;
                self.append(vec![item])?;
            }
            b'e' => {
                let items = self.to_mark()?;
                self.append(items)?;
            }
            b's' => {
                let v = self.pop()?;
                let k = self.pop()?;
                self.set(vec![k, v])?;
            }
            b'u' => {
                let items = self.to_mark()?;
                self.set(items)?;
            }
            b'c' => {
                let module = self.line()?;
                let name = self.line()?;
                self.push(Value::Global(global(module, name)?));
            }
            0x93 => {
                let (Value::Str(name), Value::Str(module)) = (self.pop()?.v, self.pop()?.v) else {
                    return Err(bad("a stack global names non-strings"));
                };
                self.push(Value::Global(global(&module, &name)?));
            }
            b'R' => self.reduce()?,
            b'Q' => self.persistent()?,
            b'b' => {
                self.pop()?;
                if !matches!(self.top()?.v, Value::Dict(_)) {
                    return Err(refused("BUILD on an object"));
                }
            }
            op => return Err(refused(format_args!("opcode {op:#04x}"))),
        }
        Ok(None)
    }
}

/// A little-endian two's complement integer of at most 8 bytes.
fn long(bytes: &[u8]) -> Result<i64, Error> {
    if bytes.len() > 8 {
        return Err(bad("an integer wider than 64 bits"));
    }
    let fill = if bytes.last().is_some_and(|b| b & 0x80 != 0) {
        0xff
    } else {
        0
    };
    let mut wide = [fill; 8];
    wide[..bytes.len()].copy_from_slice(bytes);
    Ok(i64::from_le_bytes(wide))
}

fn count_of(v: &Value) -> Result<u64, Error> {
    match v {
        Value::Int(n) if *n >= 0 => Ok(*n as u64),
        _ => Err(bad("a size is not a whole number")),
    }
}

fn counts(v: &Value) -> Result<Vec<u64>, Error> {
    let Value::Tuple(items) = v else {
        return Err(bad("a shape is not a tuple"));
    };
    if items.len() > MAX_RANK {
        return Err(bad("a tensor has too many axes"));
    }
    items.iter().map(count_of).collect()
}

fn rebuild(storage: &Value, offset: &Value, shape: &Value, stride: &Value) -> Result<View, Error> {
    let Value::Storage(storage) = storage else {
        return Err(bad("a tensor has no storage"));
    };
    let shape = counts(shape)?;
    let stride = counts(stride)?;
    if shape.len() != stride.len() {
        return Err(bad("shape and stride differ in rank"));
    }
    let view = View {
        storage: storage.clone(),
        offset: count_of(offset)?,
        shape: shape.iter().map(|&n| n as usize).collect(),
        stride,
    };
    match view.end() {
        Some(end) if end <= storage.count => Ok(view),
        _ => Err(bad(format_args!(
            "a view runs past storage {}",
            storage.key
        ))),
    }
}

/// Reads `bytes` as one value, calling nothing it names.
pub fn parse(bytes: &[u8]) -> Result<Value, Error> {
    if bytes.len() > MAX_PICKLE {
        return Err(bad("too large"));
    }
    let mut m = Machine {
        bytes,
        at: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
        copied: 0,
    };
    loop {
        let [op] = m.fixed()?;
        if let Some(v) = m.step(op)? {
            return Ok(v);
        }
    }
}
