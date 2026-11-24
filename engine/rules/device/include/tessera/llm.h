// Token rows, logits and sampling.

#ifndef TESSERA_CAP_LLM
#error "import <tessera/1/llm>"
#endif

#ifndef TESSERA_LLM_H
#define TESSERA_LLM_H

namespace tessera {

#if !defined(__METAL_VERSION__)

TESSERA_FN const llm_step* llm(const table* t) {
    return uses<llm_step>(t, LLM);
}

// A random u32 for out row `out`, rank-independent.
TESSERA_FN u32 draw(const table* t, const llm_step* s, u32 out, u32 n) {
    u64 x = get<const u64>(t, s->seed)[out] + 0x9E3779B97F4A7C15ull * (n + 1);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return u32(x ^ (x >> 31));
}

#endif

}  // namespace tessera

#endif  // TESSERA_LLM_H
