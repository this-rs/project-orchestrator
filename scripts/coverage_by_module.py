#!/usr/bin/env python3
"""lcov per-file/group analysis. usage: coverage_by_module.py lcov.info repo_root ignores.txt|- [files]"""
import sys,re,fnmatch,collections
lcov,root,ign=sys.argv[1],sys.argv[2].rstrip('/')+'/',sys.argv[3]
pats=[l.strip() for l in open(ign) if l.strip()] if ign!='-' else []
def glob2re(p):
    # codecov-style: ** any, * non-slash
    r='';i=0
    while i<len(p):
        if p[i:i+2]=='**': r+='.*';i+=2
        elif p[i]=='*': r+='[^/]*';i+=1
        else: r+=re.escape(p[i]);i+=1
    return re.compile('^'+r+'$')
pr=[glob2re(p) for p in pats]
files={};cur=None
for l in open(lcov):
    l=l.strip()
    if l.startswith('SF:'): cur=l[3:].replace(root,''); files[cur]=[0,0]
    elif l.startswith('LF:'): files[cur][0]=int(l[3:])
    elif l.startswith('LH:'): files[cur][1]=int(l[3:])
def grp(f):
    p=f.split('/')
    if p[0]=='src': return '/'.join(p[:2]) if len(p)>2 else 'src/(root)'
    if p[0]=='crates' or p[0]=='claude-code-sdk-rs' or p[0]=='claude-code-api': return '/'.join(p[:2])
    return p[0]
mode=sys.argv[4] if len(sys.argv)>4 else 'group'
g=collections.OrderedDict()
def tot(sel):
    return sum(files[f][0] for f in sel),sum(files[f][1] for f in sel)
allf=[f for f in files if not f.startswith('/') ]
ext=[f for f in files if f.startswith('/')]
print('external(non-repo) files excluded:',len(ext))
ign_f=[f for f in allf if any(r.match(f) for r in pr)]
gat=[f for f in allf if f not in ign_f]
def pc(a,b): return f"{100*b/a:.1f}" if a else "n/a"
if mode=='files':
    for f in sorted(allf):
        a,b=files[f]; print(f"{f}\t{a}\t{b}\t{pc(a,b)}\t{'IGN' if f in ign_f else ''}")
else:
    groups=collections.defaultdict(list)
    for f in allf: groups[grp(f)].append(f)
    print('group\tfiles\tlines\tcovered\tbrut%\tgated_lines\tgated_cov\tgated%')
    for k in sorted(groups):
        a,b=tot(groups[k]); ga,gb=tot([f for f in groups[k] if f in gat])
        print(f"{k}\t{len(groups[k])}\t{a}\t{b}\t{pc(a,b)}\t{ga}\t{gb}\t{pc(ga,gb)}")
a,b=tot(allf);ga,gb=tot(gat);ia,ib=tot(ign_f)
print(f"TOTAL brut {a} {b} {pc(a,b)} | gated {ga} {gb} {pc(ga,gb)} | ignored {len(ign_f)} files {ia} {ib} {pc(ia,ib)}")
