use crate::HEADERS;
use crate::layout::*;
use std::collections::HashMap;

/// A header struct: its fields, then its `alignas`.
type Declared = (Vec<(String, String, usize)>, usize);

#[derive(Default)]
struct Headers {
    numbers: HashMap<String, usize>,
    structs: HashMap<String, Declared>,
}

impl Headers {
    fn number(&self, word: &str) -> usize {
        let word = word.trim();
        (word
            .parse()
            .ok()
            .or_else(|| self.numbers.get(word).copied()))
        .unwrap_or_else(|| panic!("no number {word:?}"))
    }

    /// The element count of `a[X][Y * Z]`, and the bare name.
    fn field<'a>(&self, decl: &'a str) -> (&'a str, usize) {
        let (name, dims) = decl.split_once('[').unwrap_or((decl, ""));
        let mut count = 1;
        for dim in dims.split(['[', ']']).filter(|d| !d.trim().is_empty()) {
            count *= dim.split('*').map(|w| self.number(w)).product::<usize>();
        }
        (name.trim(), count)
    }

    fn read(&mut self, text: &str) {
        let mut open: Option<(String, Declared)> = None;
        let mut depth = 0;
        for line in text.lines().map(str::trim) {
            let number = line
                .trim_start_matches("constexpr u32 ")
                .trim_end_matches([';', ',']);
            if let Some((name, value)) = number.split_once(" = ")
                && let Ok(value) = value.parse()
            {
                self.numbers.insert(name.into(), value);
            }
            if let Some(head) = line
                .strip_prefix("struct ")
                .and_then(|l| l.strip_suffix(" {"))
            {
                let align = head
                    .split_once("alignas(")
                    .map_or(0, |(_, a)| self.number(&a[..a.find(')').unwrap()]));
                let name = head.rsplit(' ').next().unwrap();
                open = Some((name.into(), (Vec::new(), align)));
                depth = 1;
                continue;
            }
            let Some((_, (fields, _))) = open.as_mut() else {
                continue;
            };
            depth += line.matches('{').count();
            depth -= line.matches('}').count();
            if depth == 0 {
                let (name, declared) = open.take().unwrap();
                self.structs.insert(name, declared);
                continue;
            }
            if depth > 1 || line.contains('(') || line.starts_with("//") || !line.ends_with(';') {
                continue;
            }
            let (ty, names) = line.trim_end_matches(';').split_once(' ').unwrap();
            for decl in names.split(',') {
                let (name, count) = self.field(decl);
                fields.push((name.into(), ty.into(), count));
            }
        }
    }

    /// Size and alignment, by C's rules.
    fn size(&self, ty: &str) -> (usize, usize) {
        if let Some(n) = ["u8", "u16", "u32", "u64"].iter().position(|t| *t == ty) {
            return (1 << n, 1 << n);
        }
        self.layout(ty).1
    }

    fn layout(&self, name: &str) -> (Vec<Field>, (usize, usize)) {
        let (fields, align) = self
            .structs
            .get(name)
            .unwrap_or_else(|| panic!("no struct {name}"));
        let (mut at, mut most) = (0usize, (*align).max(1));
        let mut out = Vec::new();
        for (field, ty, count) in fields {
            let (size, align) = self.size(ty);
            at = at.next_multiple_of(align);
            most = most.max(align);
            out.push((leak(field), at, size * count));
            at += size * count;
        }
        (out, (at.next_multiple_of(most), most))
    }
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn check<T: Layout>(headers: &Headers) {
    let (fields, (size, align)) = headers.layout(T::C);
    assert_eq!(fields, T::FIELDS, "{} fields", T::C);
    assert_eq!(
        (size, align),
        (size_of::<T>(), align_of::<T>()),
        "{} size",
        T::C
    );
}

#[test]
fn headers_match_rust() {
    let mut headers = Headers::default();
    for h in &HEADERS {
        headers.read(h.text);
    }
    let numbers = [
        ("ARENAS", ARENAS),
        ("MAX_GROUP", MAX_GROUP),
        ("MAX_POOLS", MAX_POOLS),
        ("MAX_DEPTH", MAX_DEPTH),
        ("VALUES", VALUES),
        ("DOORBELL_SLOTS", DOORBELL_SLOTS),
        ("DOORBELL_ABORT", DOORBELL_ABORT),
        ("DOORBELL_SENT", DOORBELL_SENT),
    ];
    for (name, value) in numbers {
        assert_eq!(headers.number(name), value, "{name}");
    }
    let slots = DOORBELL_SLOTS + MAX_POOLS * MAX_DEPTH * 16;
    let sizes = SIZES + MAX_POOLS * MAX_DEPTH * MAX_GROUP * 4;
    assert!(slots <= SIZES && sizes <= DOORBELL_ABORT);
    check::<Ref>(&headers);
    check::<Item>(&headers);
    check::<Form>(&headers);
    check::<Launch>(&headers);
    check::<ConventionRef>(&headers);
    check::<Fault>(&headers);
    check::<Table>(&headers);
    check::<Barrier>(&headers);
    check::<Flag>(&headers);
    check::<Lane>(&headers);
    check::<Pool>(&headers);
    check::<Counters>(&headers);
    check::<PoolSet>(&headers);
    check::<Kept>(&headers);
    check::<KvTables>(&headers);
    check::<Request>(&headers);
    check::<Held>(&headers);
    check::<StateTables>(&headers);
    check::<LlmStep>(&headers);
    check::<ImageStep>(&headers);
}
