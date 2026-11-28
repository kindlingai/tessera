//! The pickle subset `torch.save` writes, read as data.

use crate::Error;
use std::collections::HashMap;
use std::rc::Rc;

/// The largest pickle, in bytes.
pub(crate) const MAX_PICKLE: usize = 64 << 20;
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
        view_end(self.offset, &self.shape, &self.stride)
    }
}

fn view_end(offset: u64, shape: &[usize], stride: &[u64]) -> Option<u64> {
    let mut last = offset;
    for (&n, &s) in shape.iter().zip(stride) {
        if n == 0 {
            return Some(offset);
        }
        last = last.checked_add((n as u64 - 1).checked_mul(s)?)?;
    }
    last.checked_add(1)
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

#[derive(Clone, Copy)]
struct RawStorage<'a> {
    key: &'a str,
    dtype: DType,
    count: u64,
}

impl RawStorage<'_> {
    fn into_storage(self) -> Storage {
        Storage {
            key: self.key.to_owned(),
            dtype: self.dtype,
            count: self.count,
        }
    }
}

#[derive(Clone)]
struct RawView<'a> {
    storage: RawStorage<'a>,
    offset: u64,
    shape: Vec<usize>,
    stride: Vec<u64>,
}

impl RawView<'_> {
    fn into_view(self) -> View {
        View {
            storage: self.storage.into_storage(),
            offset: self.offset,
            shape: self.shape,
            stride: self.stride,
        }
    }
}

/// Internal value keeping strings borrowed and views shared.
#[derive(Clone)]
enum Val<'a> {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(&'a str),
    Dims(u8, [u32; 4]),
    List(Vec<Val<'a>>),
    Tuple(Vec<Val<'a>>),
    Dict(Vec<(Val<'a>, Val<'a>)>),
    Global(Global),
    Storage(RawStorage<'a>),
    Tensor(Rc<RawView<'a>>),
}

impl Val<'_> {
    fn into_value(self) -> Value {
        match self {
            Val::None => Value::None,
            Val::Bool(b) => Value::Bool(b),
            Val::Int(n) => Value::Int(n),
            Val::Float(x) => Value::Float(x),
            Val::Text(s) => Value::Str(s.to_owned()),
            Val::Dims(n, a) => Value::Tuple(
                a[..n as usize]
                    .iter()
                    .map(|&v| Value::Int(v as i64))
                    .collect(),
            ),
            Val::List(xs) => Value::List(xs.into_iter().map(Val::into_value).collect()),
            Val::Tuple(xs) => Value::Tuple(xs.into_iter().map(Val::into_value).collect()),
            Val::Dict(xs) => Value::Dict(
                xs.into_iter()
                    .map(|(k, v)| (k.into_value(), v.into_value()))
                    .collect(),
            ),
            Val::Global(g) => Value::Global(g),
            Val::Storage(s) => Value::Storage(s.into_storage()),
            Val::Tensor(v) => Value::Tensor(Rc::unwrap_or_clone(v).into_view()),
        }
    }
}

/// A value with its nesting depth and node count.
#[derive(Clone)]
struct Slot<'a> {
    v: Val<'a>,
    depth: u32,
    size: u64,
}

impl<'a> Slot<'a> {
    fn leaf(v: Val<'a>) -> Self {
        Slot {
            v,
            depth: 0,
            size: 1,
        }
    }

    /// Takes `child`'s depth and size into this container.
    fn grow(&mut self, depth: u32, size: u64) -> Result<(), Error> {
        grow_meta(&mut self.depth, &mut self.size, depth, size)
    }
}

fn grow_meta(dst_depth: &mut u32, dst_size: &mut u64, depth: u32, size: u64) -> Result<(), Error> {
    *dst_depth = (*dst_depth).max(depth + 1);
    *dst_size = dst_size.saturating_add(size);
    if *dst_depth > MAX_DEPTH {
        return Err(bad("values nest too deep"));
    }
    Ok(())
}

/// Dense vector memo with sparse fallback for arbitrary keys.
struct Memo<'a> {
    dense: Vec<Option<Slot<'a>>>,
    sparse: HashMap<u32, Slot<'a>>,
    count: usize,
}

impl<'a> Memo<'a> {
    fn with_capacity(cap: usize) -> Self {
        Self {
            dense: Vec::with_capacity(cap),
            sparse: HashMap::new(),
            count: 0,
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    fn clear(&mut self) {
        self.dense.clear();
        self.sparse.clear();
        self.count = 0;
    }

    fn insert(&mut self, key: u32, slot: Slot<'a>) {
        let idx = key as usize;
        if idx < self.dense.len() {
            if self.dense[idx].replace(slot).is_none() {
                self.count += 1;
            }
            return;
        }
        if idx <= self.dense.len().saturating_add(64) {
            self.dense.resize_with(idx, || None);
            self.dense.push(Some(slot));
            self.count += 1;
            return;
        }
        if self.sparse.insert(key, slot).is_none() {
            self.count += 1;
        }
    }

    fn get(&self, key: u32) -> Option<&Slot<'a>> {
        let idx = key as usize;
        if idx < self.dense.len() {
            self.dense[idx].as_ref()
        } else {
            self.sparse.get(&key)
        }
    }
}

struct Machine<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<Slot<'a>>,
    marks: Vec<usize>,
    memo: Memo<'a>,
    copied: u64,
}

impl<'a> Machine<'a> {
    #[inline(always)]
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len());
        let end = end.ok_or_else(|| bad("ends early"))?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    #[inline(always)]
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

    #[inline(always)]
    fn text(&mut self, n: usize) -> Result<Val<'a>, Error> {
        let s = std::str::from_utf8(self.take(n)?).map_err(bad)?;
        Ok(Val::Text(s))
    }

    #[inline(always)]
    fn pop(&mut self) -> Result<Slot<'a>, Error> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() <= floor {
            return Err(bad("the stack runs empty"));
        }
        Ok(self.stack.pop().expect("checked"))
    }

    fn top(&mut self) -> Result<&mut Slot<'a>, Error> {
        self.stack
            .last_mut()
            .ok_or_else(|| bad("the stack runs empty"))
    }

    fn pop_mark(&mut self) -> Result<usize, Error> {
        self.marks.pop().ok_or_else(|| bad("no mark"))
    }

    #[inline(always)]
    fn push(&mut self, v: Val<'a>) {
        self.stack.push(Slot::leaf(v));
    }

    fn tuple_from(&mut self, start: usize) -> Result<(), Error> {
        let count = self.stack.len() - start;
        if (1..=4).contains(&count) {
            let mut dims = [0u32; 4];
            let mut all_dims = true;
            for (i, item) in self.stack[start..].iter().enumerate() {
                match item.v {
                    Val::Int(n) if (0..=u32::MAX as i64).contains(&n) => dims[i] = n as u32,
                    _ => {
                        all_dims = false;
                        break;
                    }
                }
            }
            if all_dims {
                self.stack.truncate(start);
                self.stack.push(Slot {
                    v: Val::Dims(count as u8, dims),
                    depth: 1,
                    size: 1 + count as u64,
                });
                return Ok(());
            }
        }
        let mut slot = Slot::leaf(Val::None);
        let mut values = Vec::with_capacity(count);
        for item in self.stack.drain(start..) {
            slot.grow(item.depth, item.size)?;
            values.push(item.v);
        }
        slot.v = Val::Tuple(values);
        self.stack.push(slot);
        Ok(())
    }

    fn tuple_n(&mut self, n: usize) -> Result<(), Error> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() < floor + n {
            return Err(bad("the stack runs empty"));
        }
        let start = self.stack.len() - n;
        self.tuple_from(start)
    }

    fn put(&mut self, key: u32) -> Result<(), Error> {
        let slot = self.top()?.clone();
        self.memo.insert(key, slot);
        Ok(())
    }

    fn get(&mut self, key: u32) -> Result<(), Error> {
        let slot = self
            .memo
            .get(key)
            .ok_or_else(|| bad("a memo key is unset"))?;
        self.copied = self.copied.saturating_add(slot.size);
        if self.copied > MAX_COPIED {
            return Err(bad("memo reads copy too much"));
        }
        self.stack.push(slot.clone());
        Ok(())
    }

    /// A memo put for `q` and `r`, else a get.
    fn memo_op(&mut self, op: u8, key: u32) -> Result<(), Error> {
        match op {
            b'q' | b'r' => self.put(key),
            _ => self.get(key),
        }
    }

    /// Adds one item to the list on top.
    fn append_one(&mut self, item: Slot<'a>) -> Result<(), Error> {
        let list = self.top()?;
        list.grow(item.depth, item.size)?;
        let Val::List(values) = &mut list.v else {
            return Err(refused("appending to a non-list"));
        };
        values.push(item.v);
        Ok(())
    }

    /// Adds stack items from `mark` to the list below it.
    fn append_mark(&mut self, mark: usize) -> Result<(), Error> {
        if mark == 0 {
            return Err(bad("the stack runs empty"));
        }
        let count = self.stack.len() - mark;
        if count == 0 {
            return Ok(());
        }
        let mut list = std::mem::replace(&mut self.stack[mark - 1], Slot::leaf(Val::None));
        let mut depth = list.depth;
        let mut size = list.size;
        let Val::List(values) = &mut list.v else {
            return Err(refused("appending to a non-list"));
        };
        values.reserve(count);
        for item in self.stack.drain(mark..) {
            grow_meta(&mut depth, &mut size, item.depth, item.size)?;
            values.push(item.v);
        }
        list.depth = depth;
        list.size = size;
        self.stack[mark - 1] = list;
        Ok(())
    }

    /// Adds one key, value pair to the dict on top.
    fn set_one(&mut self, k: Slot<'a>, v: Slot<'a>) -> Result<(), Error> {
        let dict = self.top()?;
        dict.grow(k.depth.max(v.depth), k.size.saturating_add(v.size))?;
        let Val::Dict(pairs) = &mut dict.v else {
            return Err(refused("setting items on a non-dict"));
        };
        pairs.push((k.v, v.v));
        Ok(())
    }

    /// Adds pairs from `mark` into the dict below.
    fn set_mark(&mut self, mark: usize) -> Result<(), Error> {
        let count = self.stack.len() - mark;
        if !count.is_multiple_of(2) {
            return Err(bad("a key has no value"));
        }
        if mark == 0 {
            return Err(bad("the stack runs empty"));
        }
        if count == 0 {
            return Ok(());
        }
        let mut dict = std::mem::replace(&mut self.stack[mark - 1], Slot::leaf(Val::None));
        let mut depth = dict.depth;
        let mut size = dict.size;
        let Val::Dict(pairs) = &mut dict.v else {
            return Err(refused("setting items on a non-dict"));
        };
        pairs.reserve(count / 2);
        let mut items = self.stack.drain(mark..);
        while let (Some(k), Some(v)) = (items.next(), items.next()) {
            grow_meta(
                &mut depth,
                &mut size,
                k.depth.max(v.depth),
                k.size.saturating_add(v.size),
            )?;
            pairs.push((k.v, v.v));
        }
        drop(items);
        dict.depth = depth;
        dict.size = size;
        self.stack[mark - 1] = dict;
        Ok(())
    }

    /// Checks items from `mark` against the depth limit.
    fn check_tuple_depth(&self, mark: usize) -> Result<(), Error> {
        for item in &self.stack[mark..] {
            if item.depth >= MAX_DEPTH {
                return Err(bad("values nest too deep"));
            }
        }
        Ok(())
    }

    fn call_global(&mut self, call: Val<'a>, args: &[Val<'a>]) -> Result<(), Error> {
        let Val::Global(g) = call else {
            return Err(refused("a call on a non-global"));
        };
        let out = match g {
            Global::OrderedDict if args.is_empty() => Val::Dict(Vec::new()),
            Global::Tensor if (6..=7).contains(&args.len()) => {
                Val::Tensor(Rc::new(rebuild(&args[0], &args[1], &args[2], &args[3])?))
            }
            Global::Parameter if args.len() == 3 && matches!(args[0], Val::Tensor(_)) => {
                args[0].clone()
            }
            g => return Err(refused(format_args!("{g:?} with these arguments"))),
        };
        self.push(out);
        Ok(())
    }

    fn reduce(&mut self) -> Result<(), Error> {
        let args = self.pop()?;
        let call = self.pop()?;
        match args.v {
            Val::Tuple(args) => self.call_global(call.v, &args),
            Val::Dims(..) => {
                let Val::Global(g) = call.v else {
                    return Err(refused("a call on a non-global"));
                };
                Err(refused(format_args!("{g:?} with these arguments")))
            }
            _ => Err(refused("a call on a non-global")),
        }
    }

    /// Fuses `TUPLE` + `REDUCE` directly from `stack[mark..]`.
    fn reduce_mark(&mut self, mark: usize) -> Result<(), Error> {
        self.check_tuple_depth(mark)?;
        let floor = self.marks.last().copied().unwrap_or(0);
        if mark <= floor {
            return Err(bad("the stack runs empty"));
        }
        let Val::Global(g) = self.stack[mark - 1].v else {
            return Err(refused("a call on a non-global"));
        };
        let n = self.stack.len() - mark;
        let out = match g {
            Global::OrderedDict if n == 0 => Val::Dict(Vec::new()),
            Global::Tensor if (6..=7).contains(&n) => Val::Tensor(Rc::new(rebuild(
                &self.stack[mark].v,
                &self.stack[mark + 1].v,
                &self.stack[mark + 2].v,
                &self.stack[mark + 3].v,
            )?)),
            Global::Parameter if n == 3 && matches!(self.stack[mark].v, Val::Tensor(_)) => {
                std::mem::replace(&mut self.stack[mark].v, Val::None)
            }
            g => return Err(refused(format_args!("{g:?} with these arguments"))),
        };
        self.stack.truncate(mark - 1);
        self.push(out);
        Ok(())
    }

    fn persistent(&mut self) -> Result<(), Error> {
        let Val::Tuple(pid) = self.pop()?.v else {
            return Err(bad("a persistent id is not a tuple"));
        };
        if pid.len() != 5 {
            return Err(refused("this persistent id"));
        }
        let storage = parse_storage(&pid[0], &pid[1], &pid[2], &pid[4])?;
        self.push(Val::Storage(storage));
        Ok(())
    }

    /// Fuses `TUPLE` + `BINPERSID` directly from `stack[mark..]`.
    fn persistent_mark(&mut self, mark: usize) -> Result<(), Error> {
        self.check_tuple_depth(mark)?;
        if self.stack.len() - mark != 5 {
            return Err(refused("this persistent id"));
        }
        let storage = parse_storage(
            &self.stack[mark].v,
            &self.stack[mark + 1].v,
            &self.stack[mark + 2].v,
            &self.stack[mark + 4].v,
        )?;
        self.stack.truncate(mark);
        self.push(Val::Storage(storage));
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
                self.memo.clear();
                return Ok(Some(top.v.into_value()));
            }
            b'(' => {
                if self.marks.len() as u32 >= MAX_DEPTH {
                    return Err(bad("marks nest too deep"));
                }
                self.marks.push(self.stack.len());
            }
            b'}' => self.push(Val::Dict(Vec::new())),
            b']' => self.push(Val::List(Vec::new())),
            b')' => self.push(Val::Tuple(Vec::new())),
            b'N' => self.push(Val::None),
            0x88 => self.push(Val::Bool(true)),
            0x89 => self.push(Val::Bool(false)),
            b't' => {
                let mark = self.pop_mark()?;
                match self.bytes.get(self.at).copied() {
                    Some(b'Q') => {
                        self.at += 1;
                        self.persistent_mark(mark)?;
                    }
                    Some(b'R') => {
                        self.at += 1;
                        self.reduce_mark(mark)?;
                    }
                    _ => self.tuple_from(mark)?,
                }
            }
            0x85..=0x87 => self.tuple_n((op - 0x84) as usize)?,
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
                self.push(Val::Int(n));
            }
            b'M' => {
                let n = u16::from_le_bytes(self.fixed()?) as i64;
                self.push(Val::Int(n));
            }
            b'J' => {
                let n = i32::from_le_bytes(self.fixed()?) as i64;
                self.push(Val::Int(n));
            }
            0x8a => {
                let [n] = self.fixed()?;
                let int = long(self.take(n as usize)?)?;
                self.push(Val::Int(int));
            }
            b'G' => {
                let x = f64::from_be_bytes(self.fixed()?);
                self.push(Val::Float(x));
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
                self.append_one(item)?;
            }
            b'e' => {
                let mark = self.pop_mark()?;
                self.append_mark(mark)?;
            }
            b's' => {
                let v = self.pop()?;
                let k = self.pop()?;
                self.set_one(k, v)?;
            }
            b'u' => {
                let mark = self.pop_mark()?;
                self.set_mark(mark)?;
            }
            b'c' => {
                let module = self.line()?;
                let name = self.line()?;
                self.push(Val::Global(global(module, name)?));
            }
            0x93 => {
                let (Val::Text(name), Val::Text(module)) = (self.pop()?.v, self.pop()?.v) else {
                    return Err(bad("a stack global names non-strings"));
                };
                self.push(Val::Global(global(module, name)?));
            }
            b'R' => self.reduce()?,
            b'Q' => self.persistent()?,
            b'b' => {
                self.pop()?;
                if !matches!(self.top()?.v, Val::Dict(_)) {
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

fn count_of(v: &Val<'_>) -> Result<u64, Error> {
    match v {
        Val::Int(n) if *n >= 0 => Ok(*n as u64),
        _ => Err(bad("a size is not a whole number")),
    }
}

fn tuple_counts<T>(v: &Val<'_>, cast: impl Fn(u64) -> T) -> Result<Vec<T>, Error> {
    match v {
        Val::Dims(n, a) => Ok(a[..*n as usize].iter().map(|&x| cast(x as u64)).collect()),
        Val::Tuple(items) => {
            if items.len() > MAX_RANK {
                return Err(bad("a tensor has too many axes"));
            }
            items.iter().map(|i| Ok(cast(count_of(i)?))).collect()
        }
        _ => Err(bad("a shape is not a tuple")),
    }
}

fn parse_storage<'a>(
    tag: &Val<'a>,
    dtype: &Val<'a>,
    key: &Val<'a>,
    count: &Val<'a>,
) -> Result<RawStorage<'a>, Error> {
    let (Val::Text(tag), Val::Global(Global::Storage(dtype)), Val::Text(key)) = (tag, dtype, key)
    else {
        return Err(refused("this persistent id"));
    };
    let plain = key.bytes().all(|b| b.is_ascii_alphanumeric());
    if *tag != "storage" || key.is_empty() || !plain {
        return Err(refused(format_args!("storage key {key:?}")));
    }
    Ok(RawStorage {
        key,
        dtype: *dtype,
        count: count_of(count)?,
    })
}

fn rebuild<'a>(
    storage: &Val<'a>,
    offset: &Val<'a>,
    shape: &Val<'a>,
    stride: &Val<'a>,
) -> Result<RawView<'a>, Error> {
    let Val::Storage(storage) = *storage else {
        return Err(bad("a tensor has no storage"));
    };
    let shape = tuple_counts(shape, |n| n as usize)?;
    let stride = tuple_counts(stride, |n| n)?;
    if shape.len() != stride.len() {
        return Err(bad("shape and stride differ in rank"));
    }
    let offset = count_of(offset)?;
    match view_end(offset, &shape, &stride) {
        Some(end) if end <= storage.count => Ok(RawView {
            storage,
            offset,
            shape,
            stride,
        }),
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
        stack: Vec::with_capacity(64),
        marks: Vec::with_capacity(16),
        memo: Memo::with_capacity(bytes.len() / 32),
        copied: 0,
    };
    loop {
        let [op] = m.fixed()?;
        if let Some(v) = m.step(op)? {
            return Ok(v);
        }
    }
}
