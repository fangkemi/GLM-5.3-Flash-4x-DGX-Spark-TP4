#!/usr/bin/env python3
"""Bounded, synthetic workload comparison; raw results are kept per stage."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path

# Spark API and SSH tunnels are direct local connections.
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
ROOT = Path.cwd() / 'logs' / 'fp8-tune'
PROMPTS = [
    ('prose', 'Explain how to decide whether a medium-sized software project is ready for a database migration. Cover dependencies, schema compatibility, tests, rollout and rollback. Write a detailed guide of at least 1000 words.'),
    ('code', 'Write a complete Python module implementing a thread-safe bounded LRU cache with per-item TTL, an injectable clock, get, set, delete, clear and __len__, and six unittest tests. Return only code.'),
    ('json', 'Return a JSON array of 60 fictional inventory items. Each item has id, sku, name, price, stock and a list of three tags. Use distinct plausible values. No prose or code fence.'),
]

def fetch(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        raw = r.read()
        return json.loads(raw) if raw else None

def metrics(base):
    with urllib.request.urlopen(base + '/metrics', timeout=15) as r:
        raw = r.read().decode()
    out = {}
    for line in raw.splitlines():
        m = re.match(r'^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if m:
            out[m[1]] = out.get(m[1], 0) + float(m[3])
    if 'vllm:request_success_total' not in out:
        raise RuntimeError('Missing request-success metric; cannot exclude other traffic')
    return out

def idle(base, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        m = metrics(base)
        if m.get('vllm:num_requests_running', 0) == 0 and m.get('vllm:num_requests_waiting', 0) == 0:
            return m
        time.sleep(3)
    raise RuntimeError('Endpoint remained busy; refusing a contaminated comparison')

def chat(base, prompt, tokens=256, fixed=False, barrier=None):
    body = dict(model='glm-5.3-flash', messages=[dict(role='user', content=prompt)],
                temperature=0, top_p=1, max_tokens=tokens, stream=True,
                stream_options={'include_usage': True}, chat_template_kwargs={'enable_thinking': False})
    if fixed:
        body['min_tokens'] = tokens
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    if barrier:
        barrier.wait()
    start = time.monotonic()
    first = last = None
    usage = None
    parts = []
    reason = []
    finish = None
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            e = json.loads(data)
            if e.get('error'):
                raise RuntimeError(e['error'])
            if e.get('usage'):
                usage = e['usage']
            for choice in e.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                if text or thought:
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
                parts.append(text)
                reason.append(thought)
                finish = choice.get('finish_reason') or finish
    end = time.monotonic()
    if not usage or first is None:
        raise RuntimeError('Missing token usage or streamed content')
    n = usage['completion_tokens']
    text = ''.join(parts)
    if '\ufffd' in text or ''.join(reason):
        raise RuntimeError('Unexpected replacement characters or thinking output')
    if fixed and n != tokens:
        raise RuntimeError(f'Fixed-length benchmark expected {tokens}, received {n}')
    return dict(tokens=n, prompt_tokens=usage['prompt_tokens'], ttft_s=first-start,
                decode_tps=(n-1)/(last-first) if n>1 and last>first else None,
                wall_s=end-start, text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                finish_reason=finish)

def quality_cases():
    cases = []
    for a,b,op in [(47,85,'+'),(13,17,'*'),(19,23,'*'),(143,57,'-'),(144,12,'/')]:
        value = {'+':a+b, '*':a*b, '-':a-b, '/':a//b}[op]
        cases.append((f'math-{a}{op}{b}', f'Calculate {a} {op} {b}. Return ONLY a JSON object with one key "answer" and a numeric value.', {'answer':value}))
    cases += [
        ('sort', 'Sort [9, -2, 5, 5, 0, 17] in ascending order. Return only a JSON object with key "sorted".', {'sorted':[-2,0,5,5,9,17]}),
        ('filter', 'From [3, 8, 12, 5, 20, 7], select only the even numbers and sum them. Return ONLY JSON with keys "even" and "sum".', {'even':[8,12,20],'sum':40}),
        ('logic', 'A is taller than B. B is taller than C. D is shorter than C. Return ONLY JSON with key "order", listing their names from tallest to shortest.', {'order':['A','B','C','D']}),
        ('unicode', 'Return ONLY a JSON object whose "text" is exactly "北京，上海，广州" and whose "count" is 3.', {'text':'北京，上海，广州','count':3}),
        ('transform', 'Given {"red":4,"blue":7,"green":2}, increase each value by 3. Return ONLY the transformed JSON object.', {'red':7,'blue':10,'green':5}),
    ]
    # Long synthetic recall crosses the 8192-token prefill boundary; no real user data.
    rows = [f'Record {i:04d}: token={hashlib.sha256(str(i).encode()).hexdigest()[:12]}; status=archived.' for i in range(600)]
    ids = [12,301,588]
    expected = {str(i):hashlib.sha256(str(i).encode()).hexdigest()[:12] for i in ids}
    prompt = 'Use only the records below.\n'+'\n'.join(rows)+'\nReturn ONLY a JSON object mapping record numbers "12", "301", and "588" to their token strings.'
    cases.append(('long-recall',prompt,expected))
    return cases

def parse_json(text):
    text=text.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    return json.loads(text)

def semantic_json(text, expected):
    decoder=json.JSONDecoder()
    for i,ch in enumerate(text):
        if ch not in '{[':
            continue
        try:
            obj,_=decoder.raw_decode(text[i:])
            if obj==expected:
                return True
        except ValueError:
            pass
    return False

def run_quality(base, stage):
    out=[]
    for name,prompt,expected in quality_cases():
        idle(base)
        row=chat(base,prompt,tokens=160)
        try:
            parsed=parse_json(row['text'])
            passed=parsed==expected
        except (ValueError,TypeError):
            parsed=None
            passed=False
        row.update(name=name,passed=passed,strict_json_passed=passed,semantic_passed=semantic_json(row["text"],expected),expected=expected,parsed=parsed)
        out.append(row)
        print(f'{stage} quality {name}: {"PASS" if passed else "FAIL"} prompt_tokens={row["prompt_tokens"]}',flush=True)
        (ROOT/f'{stage}-quality.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out

def run_perf(base,stage,rounds=3,concurrency=1):
    rows=[]
    # Full-output warmups are excluded, including compilation and prefix-cache population.
    for kind,prompt in PROMPTS:
        idle(base)
        chat(base,prompt,fixed=True)
        print(f'{stage} warmup {kind}: done',flush=True)
    for rnd in range(rounds):
        order=PROMPTS if rnd%2==0 else list(reversed(PROMPTS))
        for kind,prompt in order:
            for attempt in range(3):
                m0=idle(base)
                if concurrency==1:
                    samples=[chat(base,prompt,fixed=True)]
                    wall=samples[0]['wall_s']
                else:
                    barrier=threading.Barrier(concurrency)
                    start=time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                        samples=list(pool.map(lambda i:chat(base,PROMPTS[i%len(PROMPTS)][1]+f'\nUse example variant {i+1}.',fixed=True,barrier=barrier),range(concurrency)))
                    wall=time.monotonic()-start
                time.sleep(.15)
                m1=metrics(base)
                successes=m1['vllm:request_success_total']-m0['vllm:request_success_total']
                foreign=successes!=concurrency
                row=dict(round=rnd,kind=kind,attempt=attempt,foreign=foreign,success_delta=successes,
                         concurrency=concurrency,wall_s=wall,aggregate_tps=sum(x['tokens'] for x in samples)/wall,
                         samples=samples,drafts=m1.get('vllm:spec_decode_num_drafts_total',0)-m0.get('vllm:spec_decode_num_drafts_total',0),
                         drafted=m1.get('vllm:spec_decode_num_draft_tokens_total',0)-m0.get('vllm:spec_decode_num_draft_tokens_total',0),
                         accepted=m1.get('vllm:spec_decode_num_accepted_tokens_total',0)-m0.get('vllm:spec_decode_num_accepted_tokens_total',0))
                rows.append(row)
                (ROOT/f'{stage}-perf.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
                print(f'{stage} round={rnd} {kind} c={concurrency}: decode={statistics.mean(x["decode_tps"] for x in samples):.2f} tok/s, TTFT={statistics.mean(x["ttft_s"] for x in samples):.3f}s, foreign={foreign}',flush=True)
                if not foreign:
                    break
            else:
                raise RuntimeError('Repeated unrelated traffic invalidated benchmark')
    return rows

def run(stage,base,rounds=3,concurrency=1,quality=True):
    q=run_quality(base,stage) if quality else []
    rows=run_perf(base,stage,rounds,concurrency)
    valid=[r for r in rows if not r['foreign']]
    summaries={}
    for kind,_ in PROMPTS:
        group=[r for r in valid if r['kind']==kind]
        samples=[s for r in group for s in r['samples']]
        summaries[kind]=dict(decode_tps=statistics.median(s['decode_tps'] for s in samples),
                             ttft_s=statistics.median(s['ttft_s'] for s in samples),
                             aggregate_tps=statistics.median(r['aggregate_tps'] for r in group),
                             acceptance=sum(r['accepted'] for r in group)/sum(r['drafted'] for r in group))
    result=dict(stage=stage,base=base,rounds=rounds,concurrency=concurrency,
                quality_passed=sum(r['passed'] for r in q),quality_total=len(q),semantic_passed=sum(r["semantic_passed"] for r in q),summary=summaries,
                median_decode_tps=statistics.median(s['decode_tps'] for r in valid for s in r['samples']),
                foreign_rows=sum(r['foreign'] for r in rows),time=time.time())
    (ROOT/f'{stage}-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage')
    p.add_argument('--base',default='http://127.0.0.1:8888')
    p.add_argument('--out',type=Path,default=ROOT)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--skip-quality',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.stage):
        p.error('stage must contain only letters, digits, underscores or hyphens')
    ROOT=a.out.resolve()
    ROOT.mkdir(parents=True,exist_ok=True)
    run(a.stage,a.base,a.rounds,a.concurrency,not a.skip_quality)
