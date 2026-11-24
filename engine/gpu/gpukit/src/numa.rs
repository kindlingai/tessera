//! Best-effort Linux NUMA discovery, CPU affinity and memory policy.

#[cfg(target_os = "linux")]
use std::ffi::{c_int, c_ulong};
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
use std::ffi::{c_long, c_void};

/// Highest CPU or NUMA node index accepted from sysfs lists.
const MAX_LIST_ID: usize = 65_536;

/// Normalizes a driver PCI bus id to sysfs BDF.
pub(crate) fn pci_bdf(raw: &str) -> Option<String> {
    let mut parts = raw.trim().split(':');
    let dom = u32::from_str_radix(parts.next()?, 16).ok()?;
    let bus = u8::from_str_radix(parts.next()?, 16).ok()?;
    let mut dev_fn = parts.next()?.split('.');
    if parts.next().is_some() {
        return None;
    }
    let dev = u8::from_str_radix(dev_fn.next()?, 16).ok()?;
    let func = u8::from_str_radix(dev_fn.next()?, 16).ok()?;
    if dev_fn.next().is_some() {
        return None;
    }
    Some(format!("{dom:04x}:{bus:02x}:{dev:02x}.{func:x}"))
}

/// Parses a sysfs `numa_node` file. Negative values return `None`.
pub(crate) fn parse_node(text: &str) -> Option<u32> {
    let value: i32 = text.trim().parse().ok()?;
    u32::try_from(value).ok()
}

/// Parses a Linux sysfs range list such as `0-15,64-79`.
pub(crate) fn parse_list(text: &str) -> Vec<usize> {
    let mut ids = Vec::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let (Ok(lo), Ok(hi)) = (a.parse::<usize>(), b.parse::<usize>()) else {
                    continue;
                };
                if lo <= hi && hi < MAX_LIST_ID {
                    ids.extend(lo..=hi);
                }
            }
            None => {
                if let Ok(id) = part.parse::<usize>()
                    && id < MAX_LIST_ID
                {
                    ids.push(id);
                }
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Narrows `current` to `node_cpus`, respecting disjoint or tighter masks.
pub(crate) fn intersect_cpus(current: &[usize], node_cpus: &[usize]) -> Option<Vec<usize>> {
    let kept: Vec<usize> = current
        .iter()
        .copied()
        .filter(|cpu| node_cpus.binary_search(cpu).is_ok())
        .collect();
    if kept.is_empty() || kept.len() == current.len() {
        return None;
    }
    Some(kept)
}

/// Returns the GPU's NUMA node on multi-node Linux hosts.
pub(crate) fn device_node(pci_bus_id: &str) -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        let online = std::fs::read_to_string("/sys/devices/system/node/online").ok()?;
        if parse_list(&online).len() <= 1 {
            return None;
        }
        let bdf = pci_bdf(pci_bus_id)?;
        let path = format!("/sys/bus/pci/devices/{bdf}/numa_node");
        parse_node(&std::fs::read_to_string(path).ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pci_bus_id;
        None
    }
}

/// Narrows this thread's CPU affinity to `node`, best effort.
pub(crate) fn bind_cpu(node: u32) {
    #[cfg(target_os = "linux")]
    linux_cpu::bind(node);
    #[cfg(not(target_os = "linux"))]
    let _ = node;
}

/// Restores default thread NUMA memory policy on drop.
pub(crate) struct PolicyGuard {
    restore: bool,
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        if self.restore {
            #[cfg(all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))]
            linux_mem::reset();
        }
    }
}

/// Prefers allocations on `node` when policy is default.
pub(crate) fn prefer(node: Option<u32>) -> PolicyGuard {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    if let Some(node) = node {
        return PolicyGuard {
            restore: linux_mem::prefer(node),
        };
    }
    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    let _ = node;
    PolicyGuard { restore: false }
}

#[cfg(target_os = "linux")]
mod linux_cpu {
    use super::{c_int, c_ulong, intersect_cpus, parse_list};

    const BITS: usize = c_ulong::BITS as usize;

    unsafe extern "C" {
        fn sched_getaffinity(pid: c_int, size: usize, mask: *mut c_ulong) -> c_int;
        fn sched_setaffinity(pid: c_int, size: usize, mask: *const c_ulong) -> c_int;
    }

    fn read_affinity() -> Option<Vec<c_ulong>> {
        let mut words = 16usize;
        while words <= 64 {
            let mut mask = vec![0 as c_ulong; words];
            let bytes = words * size_of::<c_ulong>();
            if unsafe { sched_getaffinity(0, bytes, mask.as_mut_ptr()) } == 0 {
                return Some(mask);
            }
            words *= 2;
        }
        None
    }

    pub(super) fn bind(node: u32) {
        let path = format!("/sys/devices/system/node/node{node}/cpulist");
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let node_cpus = parse_list(&text);
        if node_cpus.is_empty() {
            return;
        }
        let Some(mask) = read_affinity() else {
            return;
        };
        let mut current = Vec::new();
        for (word_idx, &word) in mask.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                current.push(word_idx * BITS + tz);
                bits &= bits - 1;
            }
        }
        let Some(kept) = intersect_cpus(&current, &node_cpus) else {
            return;
        };
        let mut next = vec![0 as c_ulong; mask.len()];
        for cpu in kept {
            let word = cpu / BITS;
            if word < next.len() {
                next[word] |= (1 as c_ulong) << (cpu % BITS);
            }
        }
        let bytes = next.len() * size_of::<c_ulong>();
        let _ = unsafe { sched_setaffinity(0, bytes, next.as_ptr()) };
    }
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod linux_mem {
    use super::{c_int, c_long, c_ulong, c_void};

    const MPOL_DEFAULT: c_int = 0;
    const MPOL_PREFERRED: c_int = 1;
    const MASK_WORDS: usize = 16;
    const BITS: usize = c_ulong::BITS as usize;
    const MAX_NODES: usize = MASK_WORDS * BITS;

    #[cfg(target_arch = "x86_64")]
    const SYS_SET_MEMPOLICY: c_long = 238;
    #[cfg(target_arch = "x86_64")]
    const SYS_GET_MEMPOLICY: c_long = 239;

    #[cfg(target_arch = "aarch64")]
    const SYS_GET_MEMPOLICY: c_long = 236;
    #[cfg(target_arch = "aarch64")]
    const SYS_SET_MEMPOLICY: c_long = 237;

    unsafe extern "C" {
        fn syscall(num: c_long, ...) -> c_long;
    }

    pub(super) fn prefer(node: u32) -> bool {
        let node = node as usize;
        if node + 1 >= MAX_NODES {
            return false;
        }
        let mut mode: c_int = 0;
        let got = unsafe {
            syscall(
                SYS_GET_MEMPOLICY,
                &mut mode as *mut c_int,
                std::ptr::null_mut::<c_ulong>(),
                0 as c_ulong,
                std::ptr::null_mut::<c_void>(),
                0 as c_ulong,
            )
        };
        if got != 0 || mode != MPOL_DEFAULT {
            return false;
        }
        let mut mask = [0 as c_ulong; MASK_WORDS];
        mask[node / BITS] = (1 as c_ulong) << (node % BITS);
        let set = unsafe {
            syscall(
                SYS_SET_MEMPOLICY,
                MPOL_PREFERRED,
                mask.as_ptr(),
                MAX_NODES as c_ulong,
            )
        };
        set == 0
    }

    pub(super) fn reset() {
        let _ = unsafe {
            syscall(
                SYS_SET_MEMPOLICY,
                MPOL_DEFAULT,
                std::ptr::null::<c_ulong>(),
                0 as c_ulong,
            )
        };
    }
}
