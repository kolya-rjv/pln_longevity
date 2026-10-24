import random
from fx import *

def states(sub, src, cap):
    """distinct (end-node, used-link-set) conclusions: what the library holds, counted up to cap (cap+1 = more)."""
    out_by = {}
    for l in sub: out_by.setdefault(l.src, []).append(l)
    seen = set(); stack = [(src, frozenset())]; total_len = 0
    while stack:
        node, used = stack.pop()
        for l in out_by.get(node, ()):
            if id(l) in used: continue
            u2 = used | {id(l)}
            key = (l.dst, l.terminal, u2)
            if key in seen: continue
            seen.add(key)
            total_len += len(u2)
            if len(seen) > cap: return len(seen), total_len
            if not l.terminal: stack.append((l.dst, u2))
    return len(seen), total_len

def graph(kind, *a):
    if kind == "chain":
        return [(f"C{i}", f"C{i+1}", "Pos", 0.9, 0.9) for i in range(a[0])], "C0"
    if kind == "negchain":
        return [(f"C{i}", f"C{i+1}", "Neg" if i % 2 else "Pos", 0.9, 0.9) for i in range(a[0])], "C0"
    if kind == "layered":
        names, links = layered(a[0], a[1], a[2], 11); return links, names[0][0]
    if kind == "digraph":
        nodes = [f"D{i}" for i in range(a[0])]
        return [(x, y, "Pos", 0.5, 0.8) for x in nodes for y in nodes if x != y], nodes[0]
    if kind == "ring_chords":
        n = a[0]; rnd = random.Random(a[2] if len(a) > 2 else 3)
        links = [(f"R{i}", f"R{(i+1)%n}", "Pos", 0.6, 0.8) for i in range(n)]
        for _ in range(a[1]):
            x, y = rnd.sample(range(n), 2); links.append((f"R{x}", f"R{y}", "Neg", 0.5, 0.7))
        return links, "R0"
