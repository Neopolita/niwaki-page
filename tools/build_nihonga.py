"""Refresh the public research record from saved results; never runs a model.

Usage: python tools/build_nihonga.py --source ../nihonga
Only explicitly selected summaries and comparison images are published.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
site = Path(__file__).resolve().parents[1]
out = site / 'nihonga'
out.mkdir(exist_ok=True)
sources = {}

def read(name):
    p = args.source / name
    sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8'))

stats = read('data/phase1/stats.json')
all_scores = read('data/phase2/scores.json')['summary']
scores = {k: all_scores[k] for k in ('bench_cn', 'bench_en', 'internal')}
efficiency = read('data/phase2/efficiency.json')
fits = read('data/phase5/bridge_fitting_pilot/summary.json')
replay = read('data/phase6/verified_student_replay/summary.json')
assessment = read('data/phase6/bridge_assessment.json')
contract = read('data/phase6/replay_contract.json')
healing = [read('data/phase6/' + folder + '/summary.json') for folder in ('healing_pilot', 'healing_pilot_lr_low')]
assert all(r['status'] == 'passed' and r['optimizer_updates'] == 120 for r in healing)
assert replay['exact_equal_controls'] == 90
candidates = [a for a in replay['aggregates'] if a['stage'] == 'all']
public = {
    'updated': '2026-09-29',
    'scope': 'Development pilots; validation reused for selection; no healed-student image-quality or speed benchmark.',
    'dataset': stats,
    'baseline': {k: {'n_images': v['n_images'], 'metrics': v['metrics']} for k,v in scores.items()},
    'baseline_efficiency': {k: efficiency[k] for k in ('gpu', 'num_denoising_steps', 'batch_size', 'per_aspect_ratio', 'overall')},
    'bridge_pretraining': [{'layer': r['layer'], 'updates': r['updates'], 'parameters': r['parameters'], 'selected_step': r['selected_step'], 'selected_validation_score': r['selected_validation_score']} for r in fits['results']],
    'student_comparisons': replay['aggregates'],
    'fresh_bridge_fit': assessment['candidate_evidence'],
    'teacher_replay': {'exact_equal_controls': replay['exact_equal_controls'], 'state_tensor_count': 301, 'legacy_max_relative_l2': replay['legacy_max_relative_l2'], 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'historical_discrepancy_cause': 'unresolved'},
    'healing': [{ 'learning_rate': lr, **{k:r[k] for k in ('status','optimizer_updates','teacher_calls','student_calls','trainable_parameters','adapter_modules','selected_step','selected_relative_error_reduction','frozen_parameter_audit_count','frozen_parameters_unchanged','peak_allocated_gib')}, 'evaluations': [{k:e[k] for k in ('step','comparisons','mean_relative_l2','max_relative_l2','mean_objective','stages')} for e in r['evaluations']]} for lr,r in zip((0.0001,0.00001),healing)],
    'source_sha256': sources,
}
(out/'results.json').write_text(json.dumps(public, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def esc(s): return html.escape(str(s), quote=True)
def bars(title, rows, maximum, unit='%', hint='lower is better'):
    items = ''.join(f'<li class="{kind}" style="--v:{value:.9f};--i:{i}"><span class="name">{esc(label)}</span><span class="track"><span class="bar"></span><span class="val">{value:.3f}{unit}</span></span></li>' for i,(label,value,kind) in enumerate(rows))
    return f'<figure class="chart"><figcaption><b>{title}</b><i>{hint}</i></figcaption><ol class="bars" style="--max:{maximum}">{items}</ol></figure>'

def table(headers, rows, caption):
    return '<div class="table-scroll" tabindex="0" role="region" aria-label="'+esc(caption)+'"><table><caption>'+caption+'</caption><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

bridge_charts = ''.join(bars(f'Block {int(a["candidate_id"][-2:])} replacement', [('Identity bypass',100*a['identity_mean_relative_l2'],'open'),('Pretrained bridge',100*a['trained_mean_relative_l2'],'solid')],15) for a in candidates)
bridge_rows = [[int(a['candidate_id'][-2:]), f"{100*a['identity_mean_relative_l2']:.3f}%", f"{100*a['trained_mean_relative_l2']:.3f}%", f"{100*a['relative_error_reduction']:.2f}%", f"{a['improved_comparisons']} / 30"] for a in candidates]
fit_rows = [[e['layer'],f"{e['fresh_validation_objective']:.6f}",f"{100*e['fresh_objective_reduction_from_identity']:.2f}%",f"{100*e['fresh_objective_relative_change']:+.4f}%"] for e in assessment['candidate_evidence']]
baseline_chart = bars('Original model · overall judge score',[(k.replace('_',' ').title(),v['metrics']['overall']['mean'],'open') for k,v in scores.items()],100,unit='',hint='0–100 · higher is better')
dataset_chart = bars('Pilot prompt capabilities',[(k.replace('_',' ').title(),v*100,'solid') for k,v in stats['dimension'].items()],30,hint='share of 20,000 prompts')
baseline_rows = [[k.replace('_',' ').title(),v['n_images'],f"{v['metrics']['overall']['mean']:.2f}",f"{v['metrics']['prompt_adherence']['mean']:.2f}",f"{v['metrics']['text_rendering']['mean']:.2f}"] for k,v in scores.items()]
healing_rows = [[e['step'],f"{100*e['mean_relative_l2']:.3f}%", f"{100*healing[1]['evaluations'][i]['mean_relative_l2']:.3f}%"] for i,e in enumerate(healing[0]['evaluations'])]

# Standalone vector figure: inspect exact point values in the accompanying table.
svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 370" role="img" aria-labelledby="title desc"><title id="title">Healing did not beat the pretrained bridge</title><desc id="desc">Mean validation velocity error at 0, 30, 60, 90 and 120 updates. Both learning rates stay above the initial 5.518 percent. Exact values are in the page table.</desc><rect width="820" height="370" fill="#F7EEE3"/><g font-family="Georgia,serif" font-size="16" fill="#231F1A">']
for y in (0,3,6,9,12):
    yy = 290-y/12*240
    svg.append(f'<path d="M65 {yy} H785" stroke="#D9CDBB"/><text x="52" y="{yy+5}" text-anchor="end">{y}%</text>')
for step in (0,30,60,90,120):
    xx=65+step/120*720
    svg.append(f'<text x="{xx}" y="316" text-anchor="middle">{step}</text>')
svg.append('<text x="425" y="345" text-anchor="middle">Optimizer updates</text><text x="65" y="25">Mean validation velocity error · lower is better</text>')
base_y=290-healing[0]['evaluations'][0]['mean_relative_l2']*100/12*240
svg.append(f'<path d="M65 {base_y} H785" stroke="#746B5D" stroke-dasharray="3 5"/><text x="785" y="{base_y+20}" text-anchor="end">Pretrained bridge · 5.518%</text>')
for report,color,dash in zip(healing,('#746B5D','#46643F'),('8 4','none')):
    points=[(65+e['step']/120*720,290-e['mean_relative_l2']*100/12*240) for e in report['evaluations']]
    svg.append('<polyline fill="none" stroke="'+color+'" stroke-width="2.5" stroke-dasharray="'+dash+'" points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+'"/>')
    svg.extend(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{color}"><title>{e["step"]} updates: {e["mean_relative_l2"]*100:.6f}%</title></circle>' for (x,y),e in zip(points,report['evaluations']))
svg.append('</g></svg>')
(out/'healing.svg').write_text(''.join(svg),encoding='utf-8')

gallery_names = [('internal-p1-5bec387a9a76','An astronaut skateboarding down a ramp at a skate park, digital painting','The teacher depicts an astronaut. The block-4 identity bypass loses the spacesuit; block 5 retains it, with a different pose and background.'),('internal-p1-e0d8dd3b84cd','Generate an image of an alpine flower meadow. Style: candid photo, natural light.','All three retain the meadow, while flowers and composition change.'),('internal-p1-4a1cdb170e44','A small fox-like creature with crystal antlers, 3D render (original prompt in Chinese).','Both bypasses preserve the broad concept; proportions, fur and pose change.'),('internal-p1-425ec4e55730','A movie still of an electric kettle on a white background.','The kettle design changes substantially; the prompt does not require the teacher’s exact design.'),('internal-p1-6177a6812fbc','A movie still of a pile of old books (original prompt in Chinese).','Stack arrangement and surface details change. Thumbnail inspection is not an aesthetic ranking.')]
(out/'images').mkdir(exist_ok=True)
gallery=''
for ident,prompt,interpretation in gallery_names:
    name=ident+'-comparison.png'
    source=args.source/'data/phase3/bypass/full_images_pilot'/name
    shutil.copy2(source,out/'images'/name)
    sources['data/phase3/bypass/full_images_pilot/'+name]=hashlib.sha256(source.read_bytes()).hexdigest()
    gallery+=f'<figure class="comparison"><a href="images/{name}"><img loading="lazy" src="images/{name}" alt="Teacher, block-4 identity bypass, and block-5 identity bypass, left to right: {esc(prompt)}"></a><figcaption><b>{esc(prompt)}</b><br>{esc(interpretation)}</figcaption></figure>'
public['source_sha256']=sources
(out/'results.json').write_text(json.dumps(public, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')

template = (site/'saikei/index.html').read_text(encoding='utf-8')
head=template[:template.index('<main>')].replace('<title>Saikei · Niwaki</title>','<title>Nihonga · Qwen Image compression research · Niwaki</title>')
start=head.index('  <meta name="description"')
end=head.index('  <meta name="twitter:card"')
head=head[:start]+'''  <meta name="description" content="Nihonga: a research log for compressing Qwen-Image-2.1 through layer pruning, learned bridges and architecture healing. Results, failures and reproducible measurements.">
  <link rel="canonical" href="https://niwakiai.com/nihonga/">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Nihonga · Qwen Image compression research">
  <meta property="og:description" content="A measured research record: learned bridges improve fixed-input predictions; two architecture-healing pilots did not improve validation.">
  <meta property="og:url" content="https://niwakiai.com/nihonga/">
  <meta property="og:image" content="https://niwakiai.com/assets/og.jpg">
'''+head[end:]
head=head.replace('<link rel="stylesheet" href="../style.css">','<link rel="stylesheet" href="../style.css">\n  <link rel="stylesheet" href="nihonga.css">')
head=head.replace('<a href="./" aria-current="page">Saikei</a>','<a href="../saikei/">Saikei</a>')
head=head.replace('      <li><a href="../nihonga/">Nihonga</a></li>\n', '')
head=head.replace('      <li><a href="../dojo/">Dojo</a></li>','      <li><a href="./" aria-current="page">Nihonga</a></li>\n      <li><a href="../dojo/">Dojo</a></li>')
footer=template[template.index('<footer>'):]
body=f'''<main id="main">
  <div class="wrap lead solo nihonga-lead">
    <p class="eyebrow">Research notebook · updated <time datetime="2026-09-29">29 September 2026</time></p>
    <h1>Nihonga <span lang="ja">日本画</span></h1>
    <p>A smaller image model, with every cut measured.</p>
    <p>We are compressing Qwen-Image-2.1: identify transformer blocks that can be replaced, learn inexpensive bridges, and distill the original model’s behavior back into the student. This is our working research record, including results that did not work.</p>
    <nav class="contents" aria-label="On this page"><a href="#foundation">Dataset &amp; baseline</a><a href="#pruning">Layer selection</a><a href="#bridges">Learned bridges</a><a href="#replay">Teacher replay</a><a href="#healing">Healing</a><a href="#record">Research record</a><a href="#status">Current result</a></nav>
  </div>
  <section class="first" id="foundation" aria-labelledby="h-foundation"><div class="wrap"><p class="eyebrow">Phases 0–2 · establish the reference</p><h2 id="h-foundation">Understand the model before removing anything</h2>
    <p class="summary">The inspected transformer has 32 blocks, a hidden width of 4,096, and approximately 7.115 billion parameters. The 60-layer examples in the original plan are illustrative; our experiments use this actual 32-block architecture. The model predicts a velocity that updates the noisy latent at each denoising step; the VAE decodes the final latent into an image.</p>
    <h3>A dataset we can grow without moving the splits</h3><p class="summary">A 100,000-prompt template pool supports a 20,000-prompt pilot, with natural-language rewrites and Chinese translations. The pilot contains {stats['by_split']['train']:,} training, {stats['by_split']['validation']:,} validation and {stats['by_split']['internal_test']:,} internal-test prompts. Counting, materials, spatial relations, text, aesthetics and other capabilities are represented. Benchmark prompts stay evaluation-only, with overlap checks before and after rewriting.</p>
    <div class="charts">{dataset_chart}{baseline_chart}</div>
    <h3>The original image model baseline</h3><p class="summary">We generated and judged 3,007 reference images: 1,000 Chinese benchmark prompts, 1,000 English benchmark prompts and 1,007 internal prompts. Another 301 paired regenerations measure generation-and-judge variation. Prompts, seeds, resolutions, scheduler settings, model revisions and judge outputs are saved.</p>
    {table(['Evaluation set','Images','Overall','Prompt adherence','Text rendering'],baseline_rows,'Original model scores · frozen local Q-Judger protocol · 0–100')}
    <p class="note">These are our local judge scores, not a claim of an official leaderboard result. Text scores cover only applicable prompts; contributing counts are included in the downloadable data. Judge calibration found an approximately 2.5-point aesthetics offset; the judge remains frozen for paired comparisons.</p>
    <dl class="facts"><div><dt>Baseline speed</dt><dd>{efficiency['overall']['latency']['mean_seconds']:.3f} seconds per image on an H100 80 GB, batch size 1, 40 denoising steps; 80 timed calls across eight aspect ratios after warm-up.</dd></div><div><dt>Baseline memory</dt><dd>{efficiency['overall']['peak_vram_allocated_bytes']/2**30:.2f} GiB peak allocated VRAM across those calls. Student inference has not yet been benchmarked under the same protocol.</dd></div></dl>
  </div></section>
  <section id="pruning" aria-labelledby="h-pruning"><div class="wrap"><p class="eyebrow">Phase 3 · measure the missing update</p><h2 id="h-pruning">Similarity is a clue; images decide what it misses</h2>
    <p class="summary">Activation similarity and temporary identity bypasses nominate blocks 2, 3, 4 and 5 for independent learned replacements. An identity bypass simply passes its input through, discarding the block’s update. It tests the cost of removing the update, not whether a cheap learned bridge could recover it.</p>
    <h3>What the first image comparisons showed</h3><p class="summary">Five prompts, one per capability, were generated with block 4 or block 5 skipped: ten bypass images. In the astronaut example, skipping block 4 loses the spacesuit. Skipping block 5 retains it. That initially favors block 5 for closer study, while all four candidates remain eligible for learned bridges.</p>
    <p class="note">Each sheet reads <b>teacher / skip block 4 / skip block 5</b>, left to right. These are identity-bypass images from Phase 3, before bridge training or healing. Teacher images came from an earlier run; unresolved cross-run variation limits causal pixel comparisons. Observations are manual, not a blinded quality study. Open any sheet at full resolution.</p>
    <div class="gallery">{gallery}</div>
    <p class="note">Pixel MAE and PSNR record changes in composition and appearance, but do not directly measure prompt adherence or aesthetic quality. None of these identity bypasses is accepted as the finished model.</p>
  </div></section>
  <section id="bridges" aria-labelledby="h-bridges"><div class="wrap"><p class="eyebrow">Phases 4–5 · recover the missing update cheaply</p><h2 id="h-bridges">A small residual bridge in place of a full block</h2>
    <div class="architecture" role="img" aria-label="Teacher: 32 transformer blocks. Student: first five blocks, rank-256 residual bridge replacing original block 5, then the 26 remaining blocks. Original slot numbers are zero-based."><div><span>Teacher</span><b>32 transformer blocks</b><small>Frozen reference</small></div><div><span>Student</span><b>Blocks 0–4 → bridge → blocks 6–31</b><small>31 original blocks + one learned replacement</small></div></div>
    <p class="summary">The bridge projects 4,096 features down to 256, applies GELU, projects back up, and adds the input: <code>h′ = h + W_up GELU(W_down h)</code>. Each bridge has 2,097,152 parameters. We fit four independent candidates, replacing one original block at a time; the four bridges are not inserted together.</p>
    <p class="summary">Bridge-only pretraining keeps the rest of the model frozen. Each candidate gets 500 updates using cached hidden states from 20 training prompts, with 10 validation prompts selecting checkpoints. The fitting objective balances prompt tokens and early, middle and late image tokens, combining normalized direction matching with a smaller relative squared-error term. Block 5’s selected checkpoint is update 300.</p>
    <h3>Does a better hidden-state fit improve the whole prediction?</h3><p class="summary">We replay each identity and trained student against the verified teacher, using the same saved inputs at three stages. Each candidate has 30 comparisons; there are 240 student forwards across four candidates and two variants.</p>
    <div class="charts">{bridge_charts}</div>
    <ul class="key" aria-label="Bridge chart legend"><li><span class="open"></span>identity bypass</li><li><span class="solid"></span>pretrained bridge</li></ul>
    {table(['Original block','Identity error','Bridge error','Relative reduction','Cases improved'],bridge_rows,'Mean validation velocity error · lower is better')}
    <p class="note"><b>How to read the percentages.</b> Velocity error is ‖student − teacher‖₂ / ‖teacher‖₂, averaged equally across prompts and stages. For block 2, 12.8788036% becomes 12.0234947%: a drop of 0.855309 percentage points. Dividing that drop by the original 12.8788036% gives a <b>6.64% relative reduction</b>. This is a fraction of prediction error removed, not an image-quality improvement. “27 / 30” counts the cases with lower error.</p>
    <p class="summary">Block 5 has the lowest mean trained error, 5.518%, and a worst case of 15.095%, so it is selected for the healing pilot. Block 4 removes a larger fraction of its identity error but ends at a slightly higher mean, 5.873%. Block 3 slightly regresses at the early and middle stages despite improving its overall mean. The small reused validation set does not establish statistical superiority.</p>
  </div></section>
  <section id="replay" aria-labelledby="h-replay"><div class="wrap"><p class="eyebrow">Phase 6 · first make the supervision repeatable</p><h2 id="h-replay">A replay discrepancy paused training</h2>
    <p class="summary">Fresh teacher predictions initially disagreed with historical saved velocities beyond our 0.1% relative-L2 tolerance. Repeating a forward in the same process was stable, and full tensor-state fingerprints matched across fresh processes. The exact cause of the historical discrepancy remains unresolved.</p>
    <h3>The numerical contract we verified</h3><p class="summary">We pinned the model revision and transformer source, H100 hardware and software versions, BF16 weights, deterministic algorithms, math attention, TF32 and reduction settings, exact input order and cache lifecycle. Each prompt gets a fresh cache; early-stage conditioning is extracted before cached middle and late forwards. Reference files are loaded after predictions are made. The audit covers 301 state tensors, including otherwise unregistered positional tables.</p>
    <p class="summary">Under this contract, all <b>90 teacher predictions match the new saved references exactly</b>. Historical targets and weights remain preserved. This establishes a working repeatable setup; it does not isolate which individual setting caused the old mismatch or prove that every setting is necessary.</p>
    <h3>Did the bridges need to be trained again?</h3><p class="summary">We captured 360 fresh block/stage hidden-state pairs for the same 30 prompts and token selections. Frozen bridge validation objectives differ from the historical objectives by less than 0.05%. The replay change alone therefore does not justify replacing the Phase 5 training results.</p>
    {table(['Block','Fresh hidden-fit objective','Reduction vs identity','Change vs historical fit'],fit_rows,'Fresh hidden-state fit check · this is a different metric from velocity error')}
    <p class="note">The hidden-fit objective combines direction and relative squared errors; it has no direct percentage interpretation. Its 38–61% reductions cannot be compared as if they were the same metric as the 3.74–17.97% velocity-error reductions.</p>
  </div></section>
  <section id="healing" aria-labelledby="h-healing"><div class="wrap"><p class="eyebrow">Phase 6 · two completed parameter-efficient trials</p><h2 id="h-healing">Training works; this recipe does not improve validation</h2>
    <p class="summary">We trained the selected bridge together with rank-8 LoRA adapters on the query, key, value and output projections of all 31 surviving attention blocks. That is 124 adapters and 10,223,616 trainable parameters. Surviving base weights stay frozen. This is a parameter-efficient implementation pilot; full-weight architecture healing is still incomplete.</p>
    <dl class="facts"><div><dt>Training</dt><dd>120 updates per run: two shuffled passes over 20 training prompts × three stages. Same initialization and sampling seeds, data, losses and validation rule; learning rates 0.0001 and 0.00001.</dd></div><div><dt>Supervision</dt><dd>Relative velocity MSE + 0.1 normalized hidden-state loss at selected surviving blocks + 0.1 bridge-output supervision. Validation prompts never contribute optimizer gradients.</dd></div><div><dt>Cache gradients</dt><dd>Early conditioning is rebuilt with current student parameters. Checkpoint recomputation uses private caches; cached-stage gradients reach prefix keys and values without changing the shared cache during backward.</dd></div><div><dt>Controls</dt><dd>Both fresh runs passed 90 exact teacher replays, reproduced all 30 baseline student predictions, passed cached-stage gradient checks, and verified 288 surviving base tensors unchanged. Checkpoints include optimizer and RNG state; results are reused on reruns.</dd></div></dl>
    <figure class="chart wide healing-chart"><figcaption><b>Healing validation trajectory</b><i>lower is better</i></figcaption><img src="healing.svg" width="820" height="370" alt="Both learning rates remain above the pretrained bridge’s 5.518% validation error across 120 updates; exact values follow."></figure>
    <ul class="line-key" aria-label="Healing graph legend"><li><span class="high-line"></span>0.0001 learning rate · dashed</li><li><span class="low-line"></span>0.00001 learning rate · solid</li><li><span class="base-line"></span>pretrained baseline · dotted</li></ul>
    {table(['Optimizer updates','Learning rate 0.0001','Learning rate 0.00001'],healing_rows,'Mean validation velocity error · same 10 prompts × three stages')}
    <p class="summary"><b>Decision: keep update 0 in both trials.</b> The pretrained bridge remains at 5.518%. At update 120, the first run reaches 7.188% and the smaller-rate run 6.113%; both are worse. Lowering the rate reduces the damage, but does not provide a successful healed checkpoint. The result is preserved rather than promoted as an improvement.</p>
    <p class="note">A “passed” run status means the execution and integrity controls passed, not that quality improved. Two rates on 20 training prompts do not isolate the reason for regression. Objective scaling, optimization and generalization still need investigation. No finished images, independent-test scores, or inference speedups have been measured for the bridge or healed student.</p>
  </div></section>
  <section id="record" aria-labelledby="h-record"><div class="wrap"><p class="eyebrow">The research record</p><h2 id="h-record">What is complete, what remains open</h2>
    <dl class="facts"><div><dt>Phase 0</dt><dd>Architecture inspected: 32 blocks and 4,096-wide hidden states.</dd></div><div><dt>Phase 1</dt><dd>20k prompt pilot serialized, with a 100k template pool and fixed split assignments.</dd></div><div><dt>Phase 2</dt><dd>Original-model images, judged baseline, repeatability analysis and H100 efficiency measurements saved.</dd></div><div><dt>Phase 3</dt><dd>Layer diagnostics and identity-bypass image pilots; blocks 2–5 retained as learned-replacement candidates.</dd></div><div><dt>Phase 4</dt><dd>Residual bottleneck bridges implemented and insertion checked.</dd></div><div><dt>Phase 5</dt><dd>Four independent bridge-pretraining pilots completed; checkpoints and optimizer states saved.</dd></div><div><dt>Phase 6</dt><dd>Repeatable teacher contract verified; fixed-input bridge comparisons and fresh hidden targets saved. Two healing pilots completed without validation improvement. Full architecture healing remains open.</dd></div><div><dt>Later phases</dt><dd>Broader healing coverage, finished-image benchmarking, progressive pruning, denoising-step distillation and optional quantization remain planned. No results are claimed for them.</dd></div></dl>
    <h3>Reproducibility and updates</h3><p class="summary">Each executable notebook step has a method description and a plain-language explanation of its result. Targets, raw outputs, metrics, selected weights, optimizer state and manifests are serialized locally, with expensive experiment artifacts backed up remotely. Matching saved steps reuse their results; changed inputs stop rather than silently overwrite the record.</p>
    <ul class="get"><li><a class="primary" href="results.json">Download the measurements<small>JSON · exact values and source SHA-256 hashes</small></a></li><li><a href="https://github.com/josejuanmartinez/nihonga">Research repository ↗<small>Phase notebooks and original plan</small></a></li><li><a href="https://github.com/Neopolita/niwaki-page/tree/main/nihonga">Page source ↗<small>Public research record</small></a></li><li><a href="healing.svg">Healing graph<small>Standalone SVG</small></a></li></ul>
    <p class="note">This page is a dated snapshot of saved experiments, not a live training dashboard. Its refresh script reads selected local summaries and copies existing comparison images; it runs no models and publishes no weights, caches or credentials. Every plotted value is available in the downloadable record.</p>
  </div></section>
  <section id="status" aria-labelledby="h-status"><div class="wrap">
    <p class="eyebrow">Current result · development pilot</p><h2 id="h-status">The bridge helps. Healing has not helped yet.</h2>
    <p class="summary">Replacing original block 5 with a pretrained bottleneck bridge reduces the student’s mean velocity error from <b>6.406% to 5.518%</b> against the original model. Both 120-update healing trials made validation worse, so we retain the pretrained bridge. Image quality and speed of this student still need to be measured.</p>
    <ul class="figures"><li><span class="n">31 <small>+ 1</small></span><span class="c">surviving transformer blocks plus one bridge, from 32 original blocks</span></li><li><span class="n">5.518<small>%</small></span><span class="c">mean fixed-input validation velocity error for the selected bridge</span></li><li><span class="n">13.86<small>%</small></span><span class="c">relative error reduction against skipping block 5</span></li><li><span class="n">90 <small>/ 90</small></span><span class="c">teacher replay predictions exactly match the verified references</span></li></ul>
    <p class="note">The validation set is 10 prompts at three denoising stages: 30 comparisons, not 30 independent prompts. It has also been used for checkpoint and candidate selection. These numbers measure predictions on saved teacher inputs, rather than completed images or independent test generalization.</p>
  </div></section>
</main>
'''
(out/'index.html').write_text(head+body+footer,encoding='utf-8')
print('Wrote Nihonga documentation, measurement snapshot, vector graph and five saved galleries.')
