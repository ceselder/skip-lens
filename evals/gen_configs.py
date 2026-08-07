import json
# sanity: peek smoke bullets
d = json.load(open('evals/results/ro_smoke.json'))
for dist in ['association', 'multihop']:
    print(f'-- {dist}[0] --')
    for b in d[dist]['0'][:4]:
        print('   ', b[:90])
# manifest: 13 raw + (L62-native AVs x jac source layers)
reg = json.load(open('evals/naive_variants.json'))
L62 = [k for k, v in reg['variants'].items() if v['native_layer'] == 62]
srcs = reg['_meta']['jlens_source_layers']
lines = [f'{k} raw 0' for k in reg['variants']]
lines += [f'{k} jac {s}' for k in L62 for s in srcs]
open('evals/configs.txt', 'w').write('\n'.join(lines) + '\n')
print(f'{len(lines)} configs | 13 raw + {len(L62)} L62-AVs x {len(srcs)} jac-srcs')
