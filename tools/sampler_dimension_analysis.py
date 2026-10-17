import json, sys, statistics as st
rows = json.load(open(sys.argv[1]))
refs = {}
data = {}
for r in rows:
    if "references" in r:
        a, b = r["references"]
        refs[r["fixture"]] = ([[(x+y)/2 for x, y in zip(ea["mean"], eb["mean"])] for ea, eb in zip(a, b)],
                              [[abs(x-y) for x, y in zip(ea["mean"], eb["mean"])] for ea, eb in zip(a, b)])
    else:
        data.setdefault((r["fixture"], r["variant"], r["paths"]), []).append(r["means"])
def err(fixture, means):
    ref, disc = refs[fixture]
    out = []
    for m, rf in zip(means, ref):
        scale = max(max(rf[:3]), 1e-6)
        out.append(max(abs(m[c]-rf[c]) for c in range(3)) / scale)
    return out
for fixture in refs:
    ref, disc = refs[fixture]
    rd = sorted(max(d[:3])/max(max(r[:3]),1e-6) for d, r in zip(disc, ref))
    print(f"\n{fixture}: {len(ref)} sites, reference half-difference p95 {rd[int(.95*len(rd))]/2:.4f}")
    print(f"{'paths':>6} " + " ".join(f"{v+' mse':>11} {v+' p95':>10}" for v in ["halton","sobol","iid"]))
    for paths in [16,32,64,128,256,512,1024,2048]:
        line = f"{paths:>6} "
        for v in ["halton","sobol","iid"]:
            es = [e for means in data[(fixture, v, paths)] for e in err(fixture, means)]
            es.sort()
            line += f"{sum(e*e for e in es)/len(es):>11.5f} {es[int(.95*len(es))]:>10.4f}"
        print(line)
print("\nmedian relative error")
for fixture in refs:
    print(fixture)
    for paths in [16,64,256,2048]:
        print(f"{paths:>6} " + " ".join(f"{v}={st.median([e for means in data[(fixture, v, paths)] for e in err(fixture, means)]):.4f}" for v in ["halton","sobol","iid"]))
