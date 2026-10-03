"""Refresh the public research record from saved results; never runs a model.

Usage: python tools/build_nihonga.py --source ../nihonga
Only explicitly selected summaries and comparison images are published.
"""
import argparse
import hashlib
import html
import json
import math
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
dataset_config = read('data/phase1/config.json')
decontamination = read('data/phase1/decontamination_report.json')
llm_benchmark_rejections = read('data/phase1/pilot_20k/llm_bench_rejections.json')
all_scores = read('data/phase2/scores.json')['summary']
scores = {k: all_scores[k] for k in ('bench_cn', 'bench_en', 'internal')}
efficiency = read('data/phase2/efficiency.json')
fits = read('data/phase5/bridge_fitting_pilot/summary.json')
replay = read('data/phase6/verified_student_replay/summary.json')
assessment = read('data/phase6/bridge_assessment.json')
contract = read('data/phase6/replay_contract.json')
healing = [read('data/phase6/' + folder + '/summary.json') for folder in ('healing_pilot', 'healing_pilot_lr_low')]
bridge_only = read('data/phase6/bridge_only_healing_v2/summary.json')
full_healing = [read('data/phase6/' + folder + '/summary.json') for folder in ('full_student_healing', 'full_student_healing_lr_low_v2')]
paired_images = read('data/phase6/paired_images/summary.json')
validation_diagnostic = read('data/phase6/validation_diagnostic.json')
expanded_validation = read('data/phase6/expanded_validation/summary.json')
expanded_trained = read('data/phase6/expanded_trained_validation_v2/summary.json')
expanded_comparison = read('data/phase6/expanded_trained_validation_v2/comparison.json')
trained_images = read('data/phase6/trained_image_pairs/summary.json')
trained_visual_review = read('data/phase6/trained_image_pairs/visual_review.json')
continuation = read('data/phase6/full_student_continuation/run_v1/summary.json')
continuation_test = read('data/phase6/full_student_continuation/test_v1/summary.json')
continuation_images = read('data/phase6/full_student_continuation/images_v1/summary.json')
update120_images = read('data/phase6/full_student_continuation/update120_images_v1/summary.json')
five_way_sheets = read('data/phase6/full_student_continuation/images_v1/five_way_sheets.json')
continuation_visual_review = read('data/phase6/full_student_continuation/visual_review.json')
text_hypothesis = read('data/phase6/full_student_continuation/text_hypothesis_v1/summary.json')
text_review = read('data/phase6/full_student_continuation/text_hypothesis_v1/review_summary.json')
text_sheets = read('data/phase6/full_student_continuation/text_hypothesis_v1/sheets.json')
text_manual = read('data/phase6/full_student_continuation/text_hypothesis_v1/manual_review.json')
text_qa = read('data/phase6/full_student_continuation/text_qa_validation_v1/summary.json')
text_qa_review = read('data/phase6/full_student_continuation/text_qa_validation_v1/review_summary.json')
text_qa_sheets = read('data/phase6/full_student_continuation/text_qa_validation_v1/labeled_sheets.json')
ALIGN = 'data/phase6/full_student_continuation/alignment_qa_v1/'
alignment_review = read(ALIGN + 'review_summary.json')
decision = read(ALIGN + 'decision_evidence.json')
decision_images = read(ALIGN + 'decision_evidence_images.json')
p7_stack = read('data/phase7/four_block_replay/summary.json')
p7_fit = read('data/phase7/region_bridge_fit/summary.json')
p7_region = read('data/phase7/region_replay/summary.json')
P7 = 'data/phase7/'
p7_heal4 = read(P7 + 'full_student_healing/summary.json')
p7_grid = {tag: read(P7 + f'stage_balanced_grid/{tag}/summary.json') for tag in ('lr_1e-6', 'lr_3e-6', 'lr_1e-5', 'lr_3e-5')}
p7_cont = read(P7 + 'stage_balanced_continuation/summary.json')
p7_capture = read(P7 + 'early_capture_v2/index.json')
p7_pool = read(P7 + 'early_pool_healing/summary.json')
p7_fit45 = read(P7 + 'region_bridge_fit_4_5/summary.json')
p7_replay45 = read(P7 + 'region_replay_4_5/summary.json')
p7_heal45 = read(P7 + 'blocks45_healing_v3/summary.json')
p7_images = {f: read(P7 + f'candidate_images/summary_{f}.json') for f in ('blocks45', 'blocks25')}
p7_review = read(P7 + 'candidate_images/visual_review.json')
assert all(r['status'] == 'passed' for r in [p7_heal4, p7_cont, p7_pool, p7_fit45, p7_replay45, p7_heal45, *p7_grid.values(), *p7_images.values()])
assert len(p7_capture['examples']) == 600 and p7_review['request_key'] == p7_images['blocks45']['request_key']
activation = read('data/phase3/activations/full/summary.json')
bypasses = [read(f'data/phase3/bypass/block_{layer:02d}_pilot/summary.json') for layer in (2,3,4,5)]
assert all(r['status'] == 'passed' and r['optimizer_updates'] == 120 for r in healing)
assert bridge_only['status'] == 'passed' and bridge_only['optimizer_updates'] == 120
assert all(r['status'] == 'passed' and r['optimizer_updates'] == 60 and r['all_student_parameters_trainable'] for r in full_healing)
assert paired_images['status'] == 'passed' and len(paired_images['rows']) == 10
assert validation_diagnostic['source_sha256'] == sources['data/phase6/full_student_healing_lr_low_v2/summary.json']
assert expanded_validation['status'] == 'passed' and expanded_validation['teacher_capture_count'] == 50
assert expanded_validation['student_forwards'] == 180 and expanded_validation['optimizer_updates'] == 0
assert expanded_trained['status'] == 'passed' and expanded_trained['student_forwards'] == 180
assert expanded_trained['optimizer_updates'] == 0
assert expanded_comparison['baseline_sha256'] == sources['data/phase6/expanded_validation/summary.json']
assert expanded_comparison['trained_sha256'] == sources['data/phase6/expanded_trained_validation_v2/summary.json']
assert trained_images['status'] == 'passed' and len(trained_images['rows']) == 11
assert len(trained_images['artifacts']) == 33 and trained_images['optimizer_updates'] == 0
assert trained_visual_review['source_id_poster'] in {r['source_id'] for r in trained_images['rows']}
assert continuation['status'] == 'passed' and continuation['optimizer_updates'] == 120
assert [e['step'] for e in continuation['evaluations']] == [60, 120, 180]
assert all(e['comparisons'] == 150 for e in continuation['evaluations'])
assert continuation_test['status'] == 'passed' and continuation_test['student_forwards'] == 240
assert continuation_images['status'] == update120_images['status'] == 'passed'
assert len(continuation_images['rows']) == len(update120_images['rows']) == 11
assert continuation_visual_review['matched_image_count'] == 55
assert len(five_way_sheets['sheets']) == 3
assert text_hypothesis['status'] == text_review['status'] == 'passed'
assert len(text_hypothesis['rows']) == 16 and len(text_hypothesis['artifacts']) == 64
assert text_review['prompt_count'] == 16 and text_review['image_count'] == 64
assert len(text_sheets['sheets']) == 4 and len(text_manual['scores']) == 16
assert text_qa['status'] == text_qa_review['status'] == 'passed'
assert text_qa_review['prompt_count'] == 45 and text_qa_review['image_count'] == 180
assert len(text_qa_sheets['sheets']) == 15
assert alignment_review['status'] == 'passed' and alignment_review['optimizer_updates'] == 0
assert alignment_review['prompt_count'] == 129 and alignment_review['image_count'] == 258
assert decision['source_sha256']['alignment_review'] == sources[ALIGN + 'review_summary.json']
assert decision['source_sha256']['image_manifest'] == sources[ALIGN + 'decision_evidence_images.json']
assert decision['decision']['selected_model'] == 'pretrained' and not decision['decision']['promote_update120']
assert p7_stack['status'] == p7_fit['status'] == p7_region['status'] == 'passed'
assert p7_stack['exact_equal_controls'] == p7_region['exact_equal_controls'] == 90
assert p7_region['student_calls'] == 150 and p7_region['optimizer_updates'] == 0
assert replay['exact_equal_controls'] == 90
candidates = [a for a in replay['aggregates'] if a['stage'] == 'all']
public = {
    'updated': '2026-10-01',
    'scope': 'Development pilots. Phase 6 closed with a blinded 129-prompt paired prompt-adherence test that kept the pretrained bridge; no independent image-quality benchmark yet.',
    'dataset': stats,
    'dataset_creation': {k:dataset_config[k] for k in ('GENERATOR_VERSION','MASTER_SEED','MASTER_POOL_SIZE','PILOT_SIZE','DIMENSION_WEIGHTS','SUBDIM_WEIGHT_OVERRIDES','LLM_MODEL','LLM_EN_REWRITE_FRACTION','LLM_ZH_TRANSLATE_FRACTION','DECON_WORD_NGRAM','DECON_CHAR_NGRAM','DECON_THRESHOLD')},
    'dataset_benchmark_overlap_check': decontamination,
    'final_llm_benchmark_rejections': len(llm_benchmark_rejections),
    'baseline': {k: {'n_images': v['n_images'], 'metrics': v['metrics']} for k,v in scores.items()},
    'baseline_efficiency': {k: efficiency[k] for k in ('gpu', 'num_denoising_steps', 'batch_size', 'per_aspect_ratio', 'overall')},
    'bridge_pretraining': [{'layer': r['layer'], 'updates': r['updates'], 'parameters': r['parameters'], 'selected_step': r['selected_step'], 'selected_validation_score': r['selected_validation_score']} for r in fits['results']],
    'student_comparisons': replay['aggregates'],
    'activation_diagnostics': {k:activation[k] for k in ('n_prompts','n_measurements','dimension_counts','difficulty_counts','aggregates','percentile_scope','scope')},
    'identity_bypass_diagnostics': [{k:r[k] for k in ('bypassed_layer','n_prompts','n_comparisons','stages','teacher_png_matches_activation_count','scope')} for r in bypasses],
    'fresh_bridge_fit': assessment['candidate_evidence'],
    'teacher_replay': {'exact_equal_controls': replay['exact_equal_controls'], 'state_tensor_count': 301, 'legacy_max_relative_l2': replay['legacy_max_relative_l2'], 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'historical_discrepancy_cause': 'unresolved'},
    'healing': [{ 'learning_rate': lr, **{k:r[k] for k in ('status','optimizer_updates','teacher_calls','student_calls','trainable_parameters','adapter_modules','selected_step','selected_relative_error_reduction','frozen_parameter_audit_count','frozen_parameters_unchanged','peak_allocated_gib')}, 'evaluations': [{k:e[k] for k in ('step','comparisons','mean_relative_l2','max_relative_l2','mean_objective','stages')} for e in r['evaluations']]} for lr,r in zip((0.0001,0.00001),healing)],
    'bridge_only_healing': {k:bridge_only[k] for k in ('status','optimizer_updates','trainable_parameters','adapter_modules','selected_step','scope')},
    'full_student_healing': [{'learning_rate': lr, **{k:r[k] for k in ('status','optimizer_updates','trainable_parameters','adapter_modules','selected_step','peak_allocated_gib','scope')}, 'evaluations': [{k:e[k] for k in ('step','comparisons','mean_relative_l2','max_relative_l2','mean_objective','stages')} for e in r['evaluations']]} for lr,r in zip((0.0001,0.000001),full_healing)],
    'paired_images': {'n_pairs':len(paired_images['rows']), 'dimensions':[r['dimension'] for r in paired_images['rows']], 'teacher_mean_seconds':sum(r['teacher']['seconds'] for r in paired_images['rows'])/len(paired_images['rows']), 'student_mean_seconds':sum(r['student']['seconds'] for r in paired_images['rows'])/len(paired_images['rows'])},
    'validation_diagnostic': validation_diagnostic,
    'expanded_validation': {k:expanded_validation[k] for k in ('status','teacher_capture_count','student_forwards','optimizer_updates','control_mean_relative_l2','new_mean_relative_l2','by_stage','by_dimension')},
    'expanded_trained_validation': {k:expanded_trained[k] for k in ('status','student_forwards','optimizer_updates','control_mean_relative_l2','new_mean_relative_l2','by_stage','by_dimension')},
    'expanded_trained_comparison': {k:expanded_comparison[k] for k in ('pretrained_mean_percent','trained_mean_percent','improved_comparisons','worsened_comparisons','by_stage','largest_regressions')},
    'trained_image_pilot': {'prompts':len(trained_images['rows']), 'images':len(trained_images['artifacts']),
                           'calls':trained_images['calls'], 'manual_review':trained_visual_review},
    'full_student_continuation': {'status':continuation['status'], 'optimizer_updates':continuation['optimizer_updates'],
        'selected_step':continuation['selected_step'], 'trainable_parameters':continuation['trainable_parameters'],
        'adapter_modules':continuation['adapter_modules'], 'scope':continuation['scope'],
        'evaluations':[{k:e[k] for k in ('step','comparisons','mean_relative_l2','max_relative_l2','stages')} for e in continuation['evaluations']]},
    'full_student_continuation_untouched_test': {'prompts':20, 'comparisons_per_model':60,
        'means':continuation_test['means'], 'optimizer_updates':continuation_test['optimizer_updates']},
    'full_student_continuation_image_review': continuation_visual_review,
    'full_student_continuation_text_hypothesis': text_review,
    'full_student_continuation_text_qa_validation': text_qa_review,
    'phase7_four_blocks': {
        'question': 'Remove blocks 2-5 at once (32 -> 28 blocks): are bridges alone enough, or is healing needed?',
        'velocity_relative_l2': [{k: a[k] for k in ('variant', 'stage', 'comparisons', 'mean_relative_l2', 'max_relative_l2', 'mean_cosine')}
                                 for a in p7_region['aggregates']],
        'region_bridge_fit': [{k: r[k] for k in ('variant', 'rank', 'parameters', 'updates', 'selected_step')} |
                              {'validation_relative_l2_by_group': {g['group']: {'skip': b['relative_l2_error'], 'bridge': g['relative_l2_error']}
                               for b, g in zip(r['baseline']['validation']['groups'], r['selected']['validation']['groups'])}}
                              for r in p7_fit['results']],
        'teacher_exact_controls': p7_region['exact_equal_controls'],
        'gate': 'mean error at most 1.0 point above the one-block student and at least 50% of the skip penalty removed',
        'conclusion': 'No bridge-only variant passes; the 28-block student needs healing.'},
    'phase7_healing_28_blocks': {
        'student': '28 blocks (2-5 removed), rank-1024 region bridge, full-student healing',
        'single_moment_lr_1e-5': [{k: e[k] for k in ('step', 'mean_relative_l2', 'gate_mean_relative_l2')} for e in p7_heal4['evaluations']],
        'learning_rate_grid': {tag: [{k: e[k] for k in ('step', 'mean_relative_l2', 'gate_mean_relative_l2')} for e in r['evaluations']] for tag, r in p7_grid.items()},
        'continuation_3e-6': [{k: e[k] for k in ('step', 'mean_relative_l2', 'gate_mean_relative_l2')} for e in p7_cont['evaluations']],
        'early_examples_captured': len(p7_capture['examples']),
        'early_pool_healing': [{k: e[k] for k in ('step', 'mean_relative_l2', 'gate_mean_relative_l2')} for e in p7_pool['evaluations']],
        'selected_step': p7_pool['selected_step']},
    'phase7_blocks_4_5': {
        'bridge_only_velocity_relative_l2': [{k: a[k] for k in ('variant', 'stage', 'mean_relative_l2')} for a in p7_replay45['aggregates']],
        'healing': [{k: e[k] for k in ('step', 'mean_relative_l2', 'gate_mean_relative_l2')} for e in p7_heal45['evaluations']],
        'selected_step': p7_heal45['selected_step'], 'gate_limit': 0.0652, 'gate_passed_steps': p7_heal45['gate_passed_steps']},
    'phase7_candidate_images': {'prompts': len(p7_images['blocks45']['rows']),
        'images': sum(len(r['artifacts']) for r in p7_images.values()), 'review': p7_review},
    'phase6_checkpoint_decision': {
        **{k: alignment_review[k] for k in ('prompt_count', 'image_count', 'optimizer_updates', 'counts', 'groups',
                                            'paired', 'text_no_clear_loss', 'decision', 'scope')},
        'evidence': decision['rows'], 'rule_checks': decision['rule_checks'],
        'examples': [{k: e[k] for k in ('source_id', 'kind', 'title', 'note', 'prompt', 'subdimension', 'scores')}
                     for e in decision['examples']],
        'per_prompt': [{k: r[k] for k in ('source_id', 'subdimension', 'language', 'prompt', 'question', 'scores')}
                       for r in alignment_review['rows']]},
    'source_sha256': sources,
}
(out/'results.json').write_text(json.dumps(public, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def esc(s): return html.escape(str(s), quote=True)
def bars(title, rows, maximum, unit='%', hint='lower is better', decimals=3):
    items = ''.join(f'<li class="{kind}" style="--v:{value:.9f};--i:{i}"><span class="name">{esc(label)}</span><span class="track"><span class="bar"></span><span class="val">{value:.{decimals}f}{unit}</span></span></li>' for i,(label,value,kind) in enumerate(rows))
    return f'<figure class="chart"><figcaption><b>{title}</b><i>{hint}</i></figcaption><ol class="bars" style="--max:{maximum}">{items}</ol></figure>'

def table(headers, rows, caption):
    return '<div class="table-scroll" tabindex="0" role="region" aria-label="'+esc(caption)+'"><table><caption>'+caption+'</caption><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

bridge_charts = ''.join(bars(f'Block {int(a["candidate_id"][-2:])} replacement', [('Identity bypass',100*a['identity_mean_relative_l2'],'open'),('Pretrained bridge',100*a['trained_mean_relative_l2'],'solid')],15) for a in candidates)
bridge_rows = [[int(a['candidate_id'][-2:]), f"{100*a['identity_mean_relative_l2']:.3f}%", f"{100*a['trained_mean_relative_l2']:.3f}%", f"{100*a['relative_error_reduction']:.2f}%", f"{a['improved_comparisons']} / 30"] for a in candidates]
fit_rows = [[e['layer'],f"{e['fresh_validation_objective']:.6f}",f"{100*e['fresh_objective_reduction_from_identity']:.2f}%",f"{100*e['fresh_objective_relative_change']:+.4f}%"] for e in assessment['candidate_evidence']]
baseline_chart = bars('Original model · overall judge score',[(k.replace('_',' ').title(),v['metrics']['overall']['mean'],'open') for k,v in scores.items()],100,unit='',hint='0–100 · higher is better')
dataset_chart = bars('Pilot prompt capabilities',[(k.replace('_',' ').title(),v*100,'solid') for k,v in stats['dimension'].items()],30,hint='share of 20,000 prompts')
split_chart = bars('Our dataset · split sizes',[(label,stats['by_split'][key],'solid' if key=='train' else 'open') for key,label in [('train','Train · fit parameters'),('validation','Validation · select checkpoints'),('internal_test','Internal test · held-out evaluation')]],20000,unit='',hint='number of prompts · 20,000 total',decimals=0)
language_chart = bars('Pilot languages',[('English',stats['language']['en']*100,'solid'),('Chinese',stats['language']['zh']*100,'solid')],100,hint='share of final pilot prompts')
source_chart = bars('How the final pilot is written',[(label,stats['source'][key]*100,'solid') for key,label in [('template','Retained template wording'),('llm_rewrite','LLM English rewrite'),('llm_translate','LLM Chinese translation')]],100,hint='share of final pilot prompts')
baseline_rows = [[k.replace('_',' ').title(),v['n_images'],f"{v['metrics']['overall']['mean']:.2f}",f"{v['metrics']['prompt_adherence']['mean']:.2f}",f"{v['metrics']['text_rendering']['mean']:.2f}"] for k,v in scores.items()]
healing_rows = [[e['step'],f"{100*e['mean_relative_l2']:.3f}%", f"{100*healing[1]['evaluations'][i]['mean_relative_l2']:.3f}%"] for i,e in enumerate(healing[0]['evaluations'])]
activation_lookup = {(r['layer_index'],r['stage']):r for r in activation['aggregates']['overall']}
activation_rows = [[layer]+[f"{activation_lookup[layer,stage]['means']['token_cosine_mean']:.6f}" for stage in ('early','middle','final')] for layer in (2,3,4,5)]
change_rows = [[layer]+[f"{100*activation_lookup[layer,stage]['means']['relative_rms_change']:.3f}%" for stage in ('early','middle','final')] for layer in (2,3,4,5)]
selection_ranking = sorted((max(activation_lookup[layer,stage]['means']['relative_rms_change'] for stage in ('early','middle','final')),layer) for layer in range(32))
assert {layer for _,layer in selection_ranking[:4]} == {2,3,4,5}
selection_rows = [[rank,layer,f'{100*value:.3f}%', 'Initial candidate' if rank<=4 else 'Comparison context'] for rank,(value,layer) in enumerate(selection_ranking[:8],start=1)]
bypass_rows = [[r['bypassed_layer']]+[f"{100*s['mean_relative_l2_error']:.3f}%" for s in r['stages']]+[f"{100*max(s['max_relative_l2_error'] for s in r['stages']):.3f}%"] for r in bypasses]
bypass_charts = ''.join(bars(stage.title()+' stage · identity bypass',[(f"Skip block {r['bypassed_layer']}",100*next(s['mean_relative_l2_error'] for s in r['stages'] if s['stage']==stage),'open') for r in bypasses],20) for stage in ('early','middle','final'))

# Display the full measured layer range; candidate details retain their precision in tables.
act_svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 390" role="img" aria-labelledby="title desc"><title id="title">Input/output activation similarity across all 32 blocks</title><desc id="desc">Mean image-token cosine across 64 prompts, at early, middle and final denoising stages. Blocks 2 through 5 are highlighted as the tested batch, not an exclusive ranking.</desc><rect width="820" height="390" fill="#F7EEE3"/><g font-family="Georgia,serif" font-size="16" fill="#231F1A">']
act_svg.append('<text x="65" y="25">Input/output cosine · closer to 1 means less directional change</text>')
act_svg.append(f'<rect x="{65+1.5/31*720:.3f}" y="50" width="{4/31*720:.3f}" height="240" fill="#A7C2A0" fill-opacity=".35"/><text x="{65+3.5/31*720:.3f}" y="365" text-anchor="middle">Tested batch: 2–5</text>')
for value in (0,.25,.5,.75,1):
    yy=290-value*240
    act_svg.append(f'<path d="M65 {yy} H785" stroke="#D9CDBB"/><text x="52" y="{yy+5}" text-anchor="end">{value:g}</text>')
for layer in (0,2,5,10,15,20,25,31):
    xx=65+layer/31*720
    act_svg.append(f'<text x="{xx}" y="316" text-anchor="middle">{layer}</text>')
act_svg.append('<text x="425" y="342" text-anchor="middle">Original transformer block · zero-based index</text>')
for stage,color,dash in [('early','#746B5D','7 4'),('middle','#8B6330','2 4'),('final','#46643F','none')]:
    points=[(65+l/31*720,290-activation_lookup[l,stage]['means']['token_cosine_mean']*240) for l in range(32)]
    act_svg.append('<polyline fill="none" stroke="'+color+'" stroke-width="2" stroke-dasharray="'+dash+'" points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+'"/>')
    for layer,(x,y) in enumerate(points):
        value=activation_lookup[layer,stage]['means']['token_cosine_mean']
        act_svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.8" fill="{color}"><title>Block {layer}, {stage}: {value:.9f}</title></circle>')
act_svg.append('</g></svg>')
(out/'activation-similarity.svg').write_text(''.join(act_svg),encoding='utf-8')

def activation_plot(filename, metric, layers, bounds, ticks, title, logarithmic=False, wide=False):
    width, height = (820,390) if wide else (520,360)
    left, right, top, bottom = 76, width-25, 60, height-90
    low, high = bounds
    transform = math.log10 if logarithmic else lambda v:v
    def px(layer): return left+(layer-layers[0])/(layers[-1]-layers[0])*(right-left)
    def py(value): return bottom-(transform(value)-transform(low))/(transform(high)-transform(low))*(bottom-top)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{esc(title)}</title><desc id="desc">Mean across 64 prompts, with early, middle and final stage curves. Blocks 2–5 are highlighted. Cosine describes direction; relative L2 measures the size of the hidden-state update. Exact candidate values are in the page tables.</desc><rect width="{width}" height="{height}" fill="#F7EEE3"/><g font-family="Georgia,serif" font-size="17" fill="#231F1A">']
    parts.append(f'<text x="{left}" y="26" font-size="17">{esc(title)}</text>')
    spacing=(right-left)/(layers[-1]-layers[0])
    parts.append(f'<rect x="{px(2)-spacing/2:.3f}" y="{top}" width="{spacing*4:.3f}" height="{bottom-top}" fill="#A7C2A0" fill-opacity=".35"/>')
    for tick in ticks:
        yy=py(tick)
        label=f'{tick:.3f}' if metric=='token_cosine_mean' else f'{tick:g}%'
        parts.append(f'<path d="M{left} {yy:.3f} H{right}" stroke="#D9CDBB"/><text x="{left-12}" y="{yy+5:.3f}" text-anchor="end">{label}</text>')
    xticks=layers if len(layers)<=6 else (0,2,5,10,15,20,25,31)
    for layer in xticks:
        parts.append(f'<text x="{px(layer):.3f}" y="{bottom+28}" text-anchor="middle">{layer}</text>')
    parts.append(f'<text x="{(left+right)/2}" y="{bottom+55}" text-anchor="middle">Original block · zero-based index</text>')
    for stage,color,dash in [('early','#746B5D','7 4'),('middle','#8B6330','2 4'),('final','#46643F','none')]:
        values=[activation_lookup[layer,stage]['means'][metric]*(100 if metric=='relative_rms_change' else 1) for layer in layers]
        assert all(low<=v<=high for v in values), (filename,stage,values)
        points=[(px(layer),py(value)) for layer,value in zip(layers,values)]
        parts.append('<polyline fill="none" stroke="'+color+'" stroke-width="2.3" stroke-dasharray="'+dash+'" points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+'"/>')
        for layer,value,(x,y) in zip(layers,values,points):
            label=f'{value:.6f}'+('%' if metric=='relative_rms_change' else '')
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{4 if len(layers)<=6 else 2.8}" fill="{color}"><title>Block {layer}, {stage}: {label}</title></circle>')
    parts.append('</g></svg>')
    (out/filename).write_text(''.join(parts),encoding='utf-8')

activation_plot('activation-cosine-zoom.svg','token_cosine_mean',list(range(1,7)),(.994,1),(.994,.996,.998,1),'Cosine · zoomed axis 0.994–1.000')
activation_plot('activation-l2-zoom.svg','relative_rms_change',list(range(1,7)),(0,18),(0,5,10,15,18),'Relative L2 change · linear axis')
activation_plot('activation-l2-all.svg','relative_rms_change',list(range(32)),(5,1500),(5,10,50,100,500,1500),'Relative L2 change · all blocks · log axis',logarithmic=True,wide=True)

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
poster = next(r for r in trained_images['rows'] if 'CRIMSON HARBOR' in r['prompt'])
for role in ('teacher', 'pretrained', 'trained'):
    source_name = 'data/phase6/trained_image_pairs/' + poster[role]['name']
    source = args.source / source_name
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != poster[role]['sha256']:
        raise RuntimeError('Matched image differs from saved report: ' + source_name)
    shutil.copy2(source, out/'images'/f'phase6-poster-{role}.png')
    sources[source_name] = actual_hash

continuation_poster = next(r for r in continuation_images['rows'] if r['source_id'] == continuation_visual_review['poster_source_id'])
update120_poster = next(r for r in update120_images['rows'] if r['source_id'] == continuation_visual_review['poster_source_id'])
for role in ('teacher', 'pretrained', 'update60', 'update120', 'update180'):
    report = update120_poster if role == 'update120' else continuation_poster
    folder = 'update120_images_v1' if role == 'update120' else 'images_v1'
    source_name = f'data/phase6/full_student_continuation/{folder}/{report[role]["name"]}'
    source = args.source / source_name
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != report[role]['sha256']:
        raise RuntimeError('Continuation poster differs from saved report: ' + source_name)
    shutil.copy2(source, out/'images'/f'phase6-continuation-poster-{role}.png')
    sources[source_name] = actual_hash
for sheet in five_way_sheets['sheets']:
    source_name = 'data/phase6/full_student_continuation/images_v1/' + sheet['name']
    source = args.source / source_name
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != sheet['sha256']:
        raise RuntimeError('Continuation sheet differs from saved report: ' + source_name)
    shutil.copy2(source, out/'images'/f'phase6-{sheet["name"]}')
    sources[source_name] = actual_hash
for sheet in text_sheets['sheets']:
    source_name = 'data/phase6/full_student_continuation/text_hypothesis_v1/' + sheet['name']
    source = args.source / source_name
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != sheet['sha256']:
        raise RuntimeError('Text comparison sheet differs from saved report: ' + source_name)
    shutil.copy2(source, out/'images'/f'phase6-{sheet["name"]}')
    sources[source_name] = actual_hash
for sheet in text_qa_sheets['sheets']:
    source_name = 'data/phase6/full_student_continuation/text_qa_validation_v1/' + sheet['name']
    source = args.source / source_name
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != sheet['sha256']:
        raise RuntimeError('Text QA sheet differs from saved report: ' + source_name)
    shutil.copy2(source, out/'images'/f'phase6-text-qa-{sheet["name"]}')
    sources[source_name] = actual_hash
for source_id in ('p1-9225ba4ccf76', 'p1-5962716b5dc8', 'p1-30b3b241cfaa'):
    row = next(r for r in text_hypothesis['rows'] if r['source_id'] == source_id)
    for role in ('teacher', 'pretrained', 'update120', 'update180'):
        source_name = 'data/phase6/full_student_continuation/text_hypothesis_v1/' + row[role]['name']
        source = args.source / source_name
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != row[role]['sha256']:
            raise RuntimeError('Text comparison PNG differs from saved report: ' + source_name)
        shutil.copy2(source, out/'images'/f'phase6-text-{source_id}-{role}.png')
        sources[source_name] = actual_hash
align_sheets, align_examples = [], []
for name, expected in sorted(decision_images['sha256'].items()):
    if name == 'decision_examples.jpg':
        continue
    source_name = ALIGN + name
    source = args.source / source_name
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != expected:
        raise RuntimeError('Alignment evidence image differs from saved manifest: ' + source_name)
    if name.startswith('labeled_sheets/'):
        target = 'phase6-align-' + name.split('/')[-1]
        align_sheets.append(target)
    else:
        target = 'phase6-align-' + name.split('/')[-1]
    shutil.copy2(source, out/'images'/target)
    sources[source_name] = actual_hash
assert len(align_sheets) == 43
public['source_sha256']=sources
(out/'results.json').write_text(json.dumps(public, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')

text_qa_options = ''.join(
    f'<option value="images/phase6-text-qa-{sheet["name"]}">Prompts {3*i+1}-{3*i+3}</option>'
    for i, sheet in enumerate(text_qa_sheets['sheets'])
)
text_qa_links = ' · '.join(
    f'<a href="images/phase6-text-qa-{sheet["name"]}">{3*i+1}-{3*i+3}</a>'
    for i, sheet in enumerate(text_qa_sheets['sheets'])
)
text_qa_first = 'images/phase6-text-qa-' + text_qa_sheets['sheets'][0]['name']

from PIL import Image as _PILImage
p7_sheet_files = []
for i in range(1, 7):
    source_name = f'data/phase7/candidate_images/sheets/sheet_{i:02d}.png'
    source = args.source / source_name
    sources[source_name] = hashlib.sha256(source.read_bytes()).hexdigest()
    target = f'phase7-candidates-sheet-{i:02d}.jpg'
    _PILImage.open(source).convert('RGB').save(out / 'images' / target, format='JPEG', quality=86, optimize=True)
    p7_sheet_files.append(target)

def sheet_viewer(slug, label, files, per, total, alt, items=None):
    # items: optional [(option label, caption html, alt text)] for galleries with one captioned image per entry.
    if items:
        options = ''.join(f'<option value="images/{f}" data-caption="{esc(c)}" data-alt="{esc(a)}">{esc(l)}</option>' for f, (l, c, a) in zip(files, items))
        links = ' · '.join(f'<a href="images/{f}">{esc(l)}</a>' for f, (l, c, a) in zip(files, items))
        first = 'images/' + files[0]
        return (f'<div class="text-qa-viewer sheet-viewer" id="{slug}-gallery" aria-label="{esc(label)}">'
                f'<div class="text-qa-controls" hidden><button type="button" class="sheet-prev">Previous</button><label for="{slug}-select">View example</label>'
                f'<select id="{slug}-select" class="sheet-select">{options}</select><button type="button" class="sheet-next">Next</button>'
                f'<span class="sheet-position" role="status" aria-live="polite">1 of {len(files)}</span></div>'
                f'<figure class="comparison"><a class="sheet-full" href="{first}"><img class="sheet-image" loading="lazy" src="{first}" alt="{esc(items[0][2])}"></a>'
                f'<figcaption class="sheet-caption">{items[0][1]}</figcaption></figure>'
                f'<details><summary>Direct links to all {len(files)} full-size images</summary><p>{links}</p></details></div>')
    options = ''.join(f'<option value="images/{f}">Prompts {per*i+1}-{min(per*i+per,total)}</option>' for i, f in enumerate(files))
    links = ' · '.join(f'<a href="images/{f}">{per*i+1}-{min(per*i+per,total)}</a>' for i, f in enumerate(files))
    first = 'images/' + files[0]
    return (f'<div class="text-qa-viewer sheet-viewer" id="{slug}-gallery" aria-label="{esc(label)}" data-per-sheet="{per}" data-total="{total}" data-alt="{esc(alt)}">'
            f'<div class="text-qa-controls" hidden><button type="button" class="sheet-prev">Previous</button><label for="{slug}-select">View sheet</label>'
            f'<select id="{slug}-select" class="sheet-select">{options}</select><button type="button" class="sheet-next">Next</button>'
            f'<span class="sheet-position" role="status" aria-live="polite">1 of {len(files)}</span></div>'
            f'<figure class="comparison"><a class="sheet-full" href="{first}"><img class="sheet-image" loading="lazy" src="{first}" alt="{esc(alt.replace("{start}", "1").replace("{end}", str(min(per, total))))}"></a>'
            f'<figcaption>Open the current sheet at full resolution.</figcaption></figure>'
            f'<details><summary>Direct links to all {len(files)} full-size sheets</summary><p>{links}</p></details></div>')

bypass_viewer = sheet_viewer('bypass', 'Phase 3 identity-bypass comparisons',
                             [ident + '-comparison.png' for ident, _, _ in gallery_names], 1, len(gallery_names), '',
                             [(prompt if len(prompt) <= 48 else prompt[:46].rstrip() + '…',
                               f'<b>{esc(prompt)}</b><br>{esc(interpretation)}',
                               f'Teacher, block-4 identity bypass, and block-5 identity bypass, left to right: {prompt}')
                              for _, prompt, interpretation in gallery_names])
text_qa_viewer = sheet_viewer('text-qa', '45 validation text prompt comparisons',
                              ['phase6-text-qa-' + s['name'] for s in text_qa_sheets['sheets']], 3, 45,
                              'Text QA prompts {start} through {end}: teacher, pretrained bridge, update 120, update 180')
five_way_viewer = sheet_viewer('five-way', '11 five-way image comparisons',
                               ['phase6-' + s['name'] for s in five_way_sheets['sheets']], 4, 11,
                               'Five-way comparisons, prompts {start} through {end}: teacher, pretrained, update 60, update 120, update 180')
text_pilot_viewer = sheet_viewer('text-pilot', '16 fresh text prompt comparisons',
                                 ['phase6-' + s['name'] for s in text_sheets['sheets']], 4, 16,
                                 'Text pilot prompts {start} through {end}: teacher, pretrained bridge, update 120, update 180')
align_viewer = sheet_viewer('align', '129 alignment prompt comparisons', align_sheets, 3, 129,
                            'Alignment prompts {start} through {end}: pretrained bridge left, update 120 right, with pass, fail or uncertain scores')

ap = alignment_review['paired']
ac = alignment_review['counts']
gain, (ci_low, ci_high) = 100*ap['clear_pass_gain_fraction'], [100*v for v in ap['paired_bootstrap_95pct']]
def gx(v): return 70 + (v + 3) / 11 * 640
decision_svg = f'''<figure class="decision-chart"><svg viewBox="0 0 780 190" role="img" aria-labelledby="decision-title decision-desc"><title id="decision-title">Update 120 gain over the pretrained bridge against the promotion bar</title><desc id="decision-desc">Paired clear-pass gain of {gain:+.1f} percentage points with a 95% bootstrap interval from {ci_low:.1f} to {ci_high:.1f}. Promotion needed at least +5 points and an interval above zero.</desc>
<g style="font-family:var(--text,Georgia,serif);font-size:15px;fill:var(--ink)">
<rect x="{gx(5):.1f}" y="30" width="{gx(8)-gx(5):.1f}" height="110" style="fill:var(--rule);fill-opacity:.55"/>
<text x="{gx(6.5):.1f}" y="52" text-anchor="middle" style="fill:var(--ink-soft)">promote zone</text>
<path d="M{gx(5):.1f} 30 V140" style="stroke:var(--ink);stroke-width:2"/><text x="{gx(5)+6:.1f}" y="134" style="fill:var(--ink)">promotion bar +5</text>
<path d="M{gx(0):.1f} 30 V140" style="stroke:var(--ink-soft);stroke-width:1.5;stroke-dasharray:5 4"/><text x="{gx(0)-6:.1f}" y="52" text-anchor="end" style="fill:var(--ink-soft)">no difference</text>
<path d="M{gx(ci_low):.1f} 92 H{gx(ci_high):.1f} M{gx(ci_low):.1f} 84 V100 M{gx(ci_high):.1f} 84 V100" style="stroke:var(--ink);stroke-width:2;fill:none"/>
<circle cx="{gx(gain):.1f}" cy="92" r="7" style="fill:var(--ink);stroke:var(--paper);stroke-width:2"><title>Gain {gain:+.1f} points; 95% CI [{ci_low:.1f}, {ci_high:.1f}]</title></circle>
<text x="{gx(gain):.1f}" y="74" text-anchor="middle" style="font-weight:600">{gain:+.1f} points</text>
<text x="{(gx(ci_low)+gx(ci_high))/2:.1f}" y="118" text-anchor="middle" style="fill:var(--ink-soft);font-size:13px">95% CI {ci_low:.1f} to {ci_high:.1f}</text>
<path d="M70 150 H710" style="stroke:var(--rule)"/>
''' + ''.join(f'<text x="{gx(t):.1f}" y="168" text-anchor="middle" style="fill:var(--ink-soft);font-size:13px">{t:+d}</text>' for t in range(-2, 9, 2)) + f'''
<text x="390" y="186" text-anchor="middle" style="fill:var(--ink-soft);font-size:13px">update 120 minus pretrained, clear-pass rate, percentage points (129 paired prompts)</text>
</g></svg><figcaption>The dot is the measured gain; the bar is its 95% paired bootstrap interval (20,000 resamples). Promotion required the dot to sit in the shaded zone <b>and</b> the interval to stay right of the dashed zero line. The dot falls short of +5 and the interval touches zero.</figcaption></figure>'''

decision_tiles = (f'<ul class="figures"><li><span class="n">{ac["pretrained"]["pass"]} <small>vs</small> {ac["update120"]["pass"]}</span><span class="c">clear passes out of 129: pretrained bridge vs update 120</span></li>'
                  f'<li><span class="n">{ap["same_clear_pass_status"]} <small>/ 129</small></span><span class="c">prompts where both models got the same result</span></li>'
                  f'<li><span class="n">{ap["update120_clear_pass_wins"]} <small>–</small> {ap["pretrained_clear_pass_wins"]}</span><span class="c">clear paired wins for update 120 vs pretrained</span></li>'
                  f'<li><span class="n">{gain:+.1f} <small>pts</small></span><span class="c">gain; 95% CI [{ci_low:.1f}, {ci_high:.1f}]; bar was +5 with CI above 0</span></li></ul>')
rule_table = table(['Predeclared promotion condition', 'Measured', 'Met?'],
                   [[c['check'], c['value'], '<b>Yes</b>' if c['met'] else '<b>No</b>'] for c in decision['rule_checks']],
                   'Update 120 needed all three conditions; the rule was saved before any image was rendered')
group_names = {'counting': 'Counting', 'spatial_relation': 'Spatial relation', 'colors': 'Colors', 'multiple_objects': 'Multiple objects'}
group_rows = []
for key, label in group_names.items():
    g = alignment_review['groups'][key]
    n = sum(g['pretrained'].values())
    group_rows.append([label, n, g['pretrained'].get('pass', 0), g['update120'].get('pass', 0),
                       f"{g['update120'].get('pass', 0) - g['pretrained'].get('pass', 0):+d}"])
group_rows.append(['<b>All</b>', 129, f"<b>{ac['pretrained']['pass']}</b>", f"<b>{ac['update120']['pass']}</b>",
                   f"<b>{ac['update120']['pass'] - ac['pretrained']['pass']:+d}</b>"])
group_table = table(['Category', 'Prompts', 'Pretrained clear passes', 'Update 120 clear passes', 'Difference'], group_rows,
                    'Clear passes by question category; uncertain images do not count as passes')
pass_chart = bars('Clear passes on 129 fresh alignment prompts', [('Pretrained bridge (selected)', ac['pretrained']['pass'], 'open'), ('Update 120 (healed)', ac['update120']['pass'], 'solid')], 129, unit='', hint='out of 129 · higher is better', decimals=0)
status_word = {'pass': 'pass', 'fail': 'fail', 'uncertain': 'uncertain'}
example_figures = ''.join(
    f'<figure class="comparison"><a href="images/phase6-align-{Path(e["name"]).name}"><img loading="lazy" src="images/phase6-align-{Path(e["name"]).name}" '
    f'alt="{esc(e["title"])}. Pretrained bridge {status_word[e["scores"]["pretrained"]]}, update 120 {status_word[e["scores"]["update120"]]}. Prompt: {esc(e["prompt"])}"></a>'
    f'<figcaption><b>{esc(e["title"])}</b> | pretrained {status_word[e["scores"]["pretrained"]]}, update 120 {status_word[e["scores"]["update120"]]}<br>{esc(e["note"])}</figcaption></figure>'
    for e in decision['examples'])
evidence_table = table(['Phase 6 comparison', 'Size', 'Pretrained bridge', 'Update 120', 'Paired result', 'Reading'],
                       [[r['test'], r['size'], r['pretrained'], r['update120'], r['paired'], r['reading']] for r in decision['rows']],
                       'Every Phase 6 comparison between the selected pretrained bridge and cumulative update 120')

p7_labels = {'one_block_bridge': ('One-block bridge (Phase 6 choice)', 31),
             'skip_blocks_2_5': ('Blocks 2-5 skipped', 28),
             'stacked_bridges_2_5': ('Four per-block bridges stacked', 28),
             'region_rank256': ('One region bridge, 2.1M parameters', 28),
             'region_rank1024': ('One region bridge, 8.4M parameters', 28)}
p7 = {(a['variant'], a['stage']): a['mean_relative_l2'] for a in p7_region['aggregates']}
p7_one, p7_skip = p7['one_block_bridge', 'all'], p7['skip_blocks_2_5', 'all']
def p7_removed(v): return 'n/a' if v == 'one_block_bridge' else f"{100*(p7_skip - p7[v, 'all'])/(p7_skip - p7_one):.0f}%"
p7_chart = bars('Velocity error against the teacher, blocks 2-5 removed', [(f'{label} · {blocks} blocks', 100*p7[v, 'all'], 'solid' if v == 'one_block_bridge' else 'open') for v, (label, blocks) in p7_labels.items()], 25, hint='mean of 10 prompts × 3 stages · lower is better')
p7_table = table(['Student', 'Blocks', 'Early', 'Middle', 'Late', 'Mean', 'Skip damage removed'],
                 [[label, blocks] + [f"{100*p7[v, st]:.2f}%" for st in ('early', 'middle', 'late')] + [f"<b>{100*p7[v, 'all']:.2f}%</b>", p7_removed(v)]
                  for v, (label, blocks) in p7_labels.items()],
                 'Velocity relative L2 error against the teacher on 10 validation prompts × 3 noise stages; all five students measured in one verified run')
p7_block_table = table(['Block removed alone (Phase 6 replay)', 'Skipped', 'With its own bridge'],
                       [[int(a['candidate_id'][-2:]), f"{100*a['identity_mean_relative_l2']:.2f}%", f"{100*a['trained_mean_relative_l2']:.2f}%"] for a in candidates],
                       'Each candidate block removed on its own; the same 10 prompts × 3 stages')
p7_group_names = {'prompt_tokens': 'Text-prompt tokens', 'early_image': 'Image tokens, early noise', 'middle_image': 'Image tokens, middle noise', 'late_image': 'Image tokens, late noise'}
p7_r256, p7_r1024 = p7_fit['results']
p7_fit_table = table(['Hidden state after block 5', 'Skip', 'Region bridge 2.1M', 'Region bridge 8.4M'],
                     [[p7_group_names[b['group']], f"{100*b['relative_l2_error']:.1f}%", f"{100*g1['relative_l2_error']:.1f}%", f"{100*g2['relative_l2_error']:.1f}%"]
                      for b, g1, g2 in zip(p7_r256['baseline']['validation']['groups'], p7_r256['selected']['validation']['groups'], p7_r1024['selected']['validation']['groups'])],
                     'Region-bridge fit on the 10 validation prompts: error of the predicted hidden state after block 5; 0% would match the teacher')

def p7_stages(evaluation, group):
    return {x['stage']: x['mean_relative_l2'] for x in evaluation[group]['stages']}
def p7_row(label, evaluation):
    m, g = p7_stages(evaluation, 'monitor'), p7_stages(evaluation, 'gate')
    return [label, f"<b>{100*evaluation['mean_relative_l2']:.2f}%</b>", f"{100*m['early']:.2f}%", f"{100*m['middle']:.2f}%", f"{100*m['late']:.2f}%",
            f"{100*evaluation['gate_mean_relative_l2']:.2f}%"]
p7_heal_headers = ['Run', 'Wider (50)', 'Early', 'Middle', 'Late', 'Check (10)']
p7_eval4 = {e['step']: e for e in p7_heal4['evaluations']}
p7_cont_eval = {e['step']: e for e in p7_cont['evaluations']}
p7_pool_eval = {e['step']: e for e in p7_pool['evaluations']}
p7_untrained = p7_grid['lr_3e-6']['evaluations'][0]
p7_heal_table = table(p7_heal_headers,
    [p7_row('Untrained', p7_untrained),
     p7_row('One moment per step, rate 1e-5, step 120', p7_eval4[120])] +
    [p7_row(f'All three moments, rate {tag[3:]}, step 60', p7_grid[tag]['evaluations'][-1]) for tag in ('lr_1e-6', 'lr_3e-6', 'lr_1e-5', 'lr_3e-5')] +
    [p7_row('Rate 3e-6 continued, step 240', p7_cont_eval[240]),
     p7_row('+ extra early examples, step 420 (selected)', p7_pool_eval[420]),
     p7_row('+ extra early examples, step 480', p7_pool_eval[480])],
    'Healing the 28-block student (blocks 2-5 removed): velocity error on the 50 wider validation prompts (with its early, middle and late moments) and on the 10 check prompts. The one-block student measures 5.39% and 5.52%')
p7_path = [(0, p7_untrained), (60, p7_grid['lr_3e-6']['evaluations'][-1])] + [(s_, p7_cont_eval[s_]) for s_ in (120, 180, 240)] + [(s_, p7_pool_eval[s_]) for s_ in (300, 360, 420, 480)]
p7_path_chart = bars('28-block student, error on the 50 wider prompts during healing', [(f'Training step {s_}', 100*e['mean_relative_l2'], 'open') for s_, e in p7_path] + [('One-block student (target)', 100*expanded_validation['new_mean_relative_l2'], 'solid')], 18, hint='steps 1-60: learning-rate grid · 61-240: three more passes · 241-480: fresh early examples · lower is better', decimals=2)
p7_45 = {(a['variant'], a['stage']): a['mean_relative_l2'] for a in p7_replay45['aggregates']}
p7_45_labels = {'one_block_bridge': ('One-block bridge (reference)', 31), 'skip_blocks_4_5': ('Blocks 4-5 skipped', 30),
                'stacked_bridges_4_5': ('Two per-block bridges', 30), 'region_rank256': ('One region bridge, 2.1M parameters', 30),
                'region_rank512': ('One region bridge, 4.2M parameters', 30)}
p7_45_skip = p7_45['skip_blocks_4_5', 'all']
def p7_45_removed(v): return 'n/a' if v == 'one_block_bridge' else f"{100*(p7_45_skip - p7_45[v, 'all'])/(p7_45_skip - p7_one):.0f}%"
p7_45_table = table(['Student', 'Blocks', 'Early', 'Middle', 'Late', 'Mean', 'Skip damage removed'],
    [[label, blocks] + [f"{100*p7_45[v, st]:.2f}%" for st in ('early', 'middle', 'late')] + [f"<b>{100*p7_45[v, 'all']:.2f}%</b>", p7_45_removed(v)]
     for v, (label, blocks) in p7_45_labels.items()],
    'Removing only blocks 4 and 5: velocity error against the teacher on the same 10 prompts × 3 stages, all five students in one verified run; the gate is 6.52%')
p7_heal45_eval = {e['step']: e for e in p7_heal45['evaluations']}
p7_heal45_table = table(p7_heal_headers,
    [p7_row('Untrained (bridge only)', p7_heal45_eval[0])] +
    [p7_row(f'Healed, step {s_}' + (' (selected)' if s_ == p7_heal45['selected_step'] else ''), p7_heal45_eval[s_]) for s_ in (60, 120, 180, 240)],
    'Healing the 30-block student (blocks 4-5 removed): rate 3e-6, all three moments per step, fresh early examples; same columns as above')
p7_models = [('teacher', 'Teacher'), ('A_healed_u60', 'A: 30 blocks, healed step 60'), ('B_healed_u240', 'B: 30 blocks, healed step 240'),
             ('C_bridge_only', 'C: 30 blocks, bridge only'), ('D_blocks25_u420', 'D: 28 blocks, healed step 420')]
p7_text_table = table(['Requested text', 'Teacher', 'A', 'B', 'C', 'D'],
    [[esc(t['requested'])] + [('pass' if t['verdicts'][m] == 'pass' else '<b>fail</b>') for m, _ in p7_models] for t in p7_review['text_prompts']] +
    [['<b>Passes</b>'] + [f"<b>{p7_review['text_pass_counts'][m]} / 5</b>" for m, _ in p7_models]],
    'Text prompts: pass only if every requested phrase is readable and correctly spelled. A, B, C: 30 blocks (healed step 60, healed step 240, bridge only); D: 28 blocks')
p7_text_failures = '; '.join(f"{label.split(':')[0]} on {esc(t['requested'])}: {esc(t['verdicts'][m][6:])}" for t in p7_review['text_prompts'] for m, label in p7_models if t['verdicts'][m] != 'pass')
p7_candidates_viewer = sheet_viewer('phase7-candidates', 'Teacher and four pruned candidates on 30 validation prompts', p7_sheet_files, 5, 30,
    'Prompts {start} through {end}: teacher, A, B, C and D, left to right')

template = (site/'saikei/index.html').read_text(encoding='utf-8')
head=template[:template.index('<main>')].replace('<title>Saikei · Niwaki</title>','<title>Nihonga · Qwen Image compression research · Niwaki</title>')
start=head.index('  <meta name="description"')
end=head.index('  <meta name="twitter:card"')
head=head[:start]+'''  <meta name="description" content="Nihonga: a research log for compressing Qwen-Image-2.1 through layer pruning, learned bridges and architecture healing. Results, failures and reproducible measurements.">
  <link rel="canonical" href="https://niwakiai.com/nihonga/">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Nihonga · Qwen Image compression research">
  <meta property="og:description" content="Phase 6 closes: full-student healing was safe but not measurably better in a blinded 129-prompt adherence test, so the pretrained bridge stays.">
  <meta property="og:url" content="https://niwakiai.com/nihonga/">
  <meta property="og:image" content="https://niwakiai.com/assets/nihonga.jpg">
'''+head[end:]
head=head.replace('<link rel="stylesheet" href="../style.css">','<link rel="stylesheet" href="../style.css">\n  <link rel="stylesheet" href="nihonga.css">')
head=head.replace('<a href="./" aria-current="page">Saikei</a>','<a href="../saikei/">Saikei</a>')
head=head.replace('      <li><a href="../nihonga/">Nihonga</a></li>\n', '')
head=head.replace('      <li><a href="../dojo/">Dojo</a></li>','      <li><a href="./" aria-current="page">Nihonga</a></li>\n      <li><a href="../dojo/">Dojo</a></li>')
footer=template[template.index('<footer>'):]
body=f'''<main id="main">
  <figure class="plate">
    <picture>
      <source srcset="../assets/nihonga.webp" type="image/webp">
      <img src="../assets/nihonga.jpg" width="1774" height="887" fetchpriority="high" alt="Engraving of a Japanese painter’s low worktable with an unfurled pine-and-mountain landscape, brushes, an inkstone and bowls of mineral pigment, under the word Nihonga.">
    </picture>
  </figure>
  <div class="wrap lead solo nihonga-lead">
    <p class="eyebrow">Research notebook | updated <time datetime="2026-10-03">3 October 2026</time></p>
    <h1>Nihonga <span lang="ja">日本画</span></h1>
    <p>A smaller image model, with every cut measured.</p>
    <p>We are compressing Qwen-Image-2.1: identify transformer blocks that can be replaced, learn inexpensive bridges, and distill the original model’s behavior back into the student. This is our working research record, including results that did not work.</p>
    <nav class="contents" aria-label="On this page"><a href="#foundation">Dataset &amp; baseline</a><a href="#pruning">Layer selection</a><a href="#bridges">Learned bridges</a><a href="#replay">Teacher replay</a><a href="#healing">Healing</a><a href="#phase7">Fewer blocks</a><a href="#record">Research record</a><a href="#status">Current result</a></nav>
  </div>
  <section class="first" id="foundation" aria-labelledby="h-foundation"><div class="wrap"><p class="eyebrow">Phases 0–2 · establish the reference</p><h2 id="h-foundation">Understand the model before removing anything</h2>
    <p class="summary">The inspected transformer has 32 blocks, a hidden width of 4,096, and approximately 7.115 billion parameters. The 60-layer examples in the original plan are illustrative; our experiments use this actual 32-block architecture. The model predicts a velocity that updates the noisy latent at each denoising step; the VAE decodes the final latent into an image.</p>
    <h3>What “internal” means</h3><p class="summary"><b>Internal means our own synthetic prompt dataset</b>, built for this compression project. It is separate from Qwen-Image-Bench. The <b>“Internal” baseline row is specifically its 1,007-prompt held-out internal-test split</b>, not the training set and not all 20,000 pilot prompts. Qwen-Image-Bench supplies the other two rows, with the same 1,000 benchmark items evaluated in Chinese and English.</p>
    <h3>Why build our own dataset?</h3><p class="summary">We need prompts to train the bridges and heal the student without training on benchmark questions. We also need validation prompts for choosing checkpoints and a separate test set for evaluating changes. A broad synthetic corpus lets us deliberately cover counting, spatial relations, text rendering, materials, scenes and visual styles instead of relying on whichever examples happen to be easy. For distillation, the original image model supplies hidden states and velocity targets; real-image training pairs are not required for this stage.</p>
    <h3>How we created it</h3>
    <dl class="facts"><div><dt>1 · structured prompts</dt><dd>Generate a 100,000-prompt template pool across five capability groups and {len(stats['subdimension'])} subdimensions. Combine objects, counts, colors, relations, materials and scenes with difficulty levels 1–5, styles, prompt formats and aspect ratios. Exact normalized duplicate prompts are removed. Counting and spatial relations receive extra weight, and text rendering receives twice its unadjusted subdimension weight.</dd></div><div><dt>2 · stable partitions</dt><dd>Assign deterministic IDs and seeds, stratify within capability/subdimension/difficulty, then select a 20,000-prompt prefix by a saved pool rank. The intended split is approximately 90% train, 5% validation and 5% internal test. The actual pilot sizes are 17,991 / 1,002 / 1,007. Split assignments remain fixed when growing the pool.</dd></div><div><dt>3 · natural language</dt><dd>Use Qwen3-8B through vLLM on Modal to rewrite selected English prompts and translate about 10% into Chinese. Pure templates can cover a narrow set of sentence shapes; natural wording and Chinese prompts broaden the conditioning the student sees. Automatic checks try to preserve quoted text and key constraints; failed rewrites fall back to the original template. Raw responses are retained.</dd></div><div><dt>4 · benchmark separation</dt><dd>Check normalized exact matches and English 6-word / Chinese 8-character n-gram overlap against a pinned Qwen-Image-Bench prompt snapshot, both before and after rewriting. At the configured 0.5 overlap threshold, the template-pool check removed zero prompts; no final rewrites were rejected for benchmark overlap. These checks reduce detectable contamination, but do not prove every prompt is semantically unrelated.</dd></div><div><dt>5 · serialize the record</dt><dd>Save each prompt’s capability, subdimension, facet, difficulty, style, aspect ratio, language, source, original template, ID, seed and split. Save the generation configuration, raw LLM outputs, overlap checks, distributions and artifact hashes. Rerunning reads the existing results rather than generating the corpus again.</dd></div></dl>
    <div class="charts">{split_chart}{dataset_chart}{source_chart}{language_chart}</div>
    <p class="note">The 100,000-prompt master pool is the template reservoir. Natural-language rewriting and translation were completed for the 20,000-prompt pilot; we have not trained on all 100,000 prompts. The bars show the actual pilot distributions from the saved Phase 1 statistics, not planned quotas. Wording checks are automatic rather than a guarantee of perfect constraint preservation.</p>
    <h3>A saved internal-test example</h3><blockquote class="dataset-example"><p>Create a photorealistic café chalkboard displaying “Today’s special: tomato soup and grilled cheese” in legible chalk writing, soft shadows, mysterious atmosphere.</p></blockquote>
    <p class="note">This held-out example tests long text rendering: creative generation, difficulty 5, photographic style, square aspect ratio, English LLM rewrite. Its seed and original template are saved. It is a test prompt, not a bridge-training example.</p>
    {table(['Split / benchmark','Role','How used so far'],[
      ['Train · 17,991 prompts','Fit model parameters','20 selected prompts in the bridge and healing pilots'],
      ['Validation · 1,002 prompts','Choose candidates and checkpoints','10 selected prompts, reused across the small pilots'],
      ['Internal test · 1,007 prompts','Our held-out evaluation set','All baseline images; subsets for Phase 3 diagnostics'],
      ['Qwen-Image-Bench · 1,000 bilingual items','External benchmark · evaluation only','1,000 Chinese and 1,000 English baseline images']
    ],'“Internal” in the baseline is the held-out test row; validation is a different split')}
    <h3>The original image model baseline</h3><p class="summary">We generated and judged 3,007 reference images: 1,000 Chinese benchmark prompts, 1,000 English benchmark prompts and 1,007 internal prompts. Another 301 paired regenerations measure generation-and-judge variation. Prompts, seeds, resolutions, scheduler settings, model revisions and judge outputs are saved.</p>
    <div class="charts">{baseline_chart}</div>
    {table(['Evaluation set','Images','Overall','Prompt adherence','Text rendering'],baseline_rows,'Original model scores · frozen local Q-Judger protocol · 0–100')}
    <p class="note">These are our local judge scores, not a claim of an official leaderboard result. Text scores cover only applicable prompts; contributing counts are included in the downloadable data. Judge calibration found an approximately 2.5-point aesthetics offset; the judge remains frozen for paired comparisons.</p>
    <dl class="facts"><div><dt>Baseline speed</dt><dd>{efficiency['overall']['latency']['mean_seconds']:.3f} seconds per image on an H100 80 GB, batch size 1, 40 denoising steps; 80 timed calls across eight aspect ratios after warm-up.</dd></div><div><dt>Baseline memory</dt><dd>{efficiency['overall']['peak_vram_allocated_bytes']/2**30:.2f} GiB peak allocated VRAM across those calls. Student inference has not yet been benchmarked under the same protocol.</dd></div></dl>
  </div></section>
  <section id="pruning" aria-labelledby="h-pruning"><div class="wrap"><p class="eyebrow">Phase 3 · measure the missing update</p><h2 id="h-pruning">Similarity is a clue; images decide what it misses</h2>
    <p class="summary"><b>We measured all 32 transformer blocks before choosing which ones to replace.</b> On 64 prompts, at three denoising stages, we compared the hidden states entering and leaving every block. Blocks 2, 3, 4 and 5 had the smallest consistent changes, so we chose them as bridge candidates. The choice came from the measurements.</p>
    <p class="summary">To make that choice, we took each block’s average relative L2 change at each of the three stages, kept the <b>largest of those three averages</b>, and ranked all 32 blocks from smallest to largest. The first four were blocks <b>2, 4, 5 and 3</b>. After selecting them, we temporarily skipped each of those four blocks separately to measure how much that changed the model’s velocity prediction. The all-block activation comparison and the four-candidate bypass tests are two different experiments.</p>
    <h3>The initial candidate ranking</h3>
    {table(['Rank','Original block','Largest stage-average L2 change','Decision'],selection_rows,'Initial shortlist · rank all 32 blocks by max(early mean, middle mean, final mean) · lower is smaller')}
    <p class="note">This rule favors blocks with consistently small updates across the sampled denoising stages, rather than a block that appears unimportant at only one stage. It is a ranking of <b>local hidden-state changes</b>, not yet bypass velocity errors or image quality. “Largest stage average” means the largest of three means over 64 prompts; it is not the worst individual prompt. The initial rank order is 2, 4, 5, 3; written in model order, the selected set is 2–5.</p>
    <h3>The activation measurements</h3><p class="summary">We measured all 32 blocks on <b>64 fixed internal prompts</b> at denoising steps <b>0, 19 and 39</b> of the 40-step teacher: <b>6,144 block/stage observations</b>. Hooks compare image-token states immediately before and after each block, excluding the text prefix. Each cosine is averaged over image tokens, then across prompts. The table shows the four tested candidates; the graph and downloadable record cover every block.</p>
    <div class="charts activation-zoom">
      <figure class="chart"><figcaption><b>Direction · cosine, zoomed</b><i>closer to 1 means less rotation</i></figcaption><a href="activation-cosine-zoom.svg"><img src="activation-cosine-zoom.svg" width="520" height="360" alt="Cosine for blocks 1–6, with a zoomed vertical axis from 0.994 to 1.000. Blocks 2–5 are highlighted. Open the SVG for a larger view."></a></figure>
      <figure class="chart"><figcaption><b>Update size · relative L2</b><i>smaller means a smaller update</i></figcaption><a href="activation-l2-zoom.svg"><img src="activation-l2-zoom.svg" width="520" height="360" alt="Relative L2 hidden-state change for blocks 1–6, on a linear vertical axis from 0% to 18%. Blocks 2–5 are highlighted. Open the SVG for a larger view."></a></figure>
    </div>
    <ul class="line-key" aria-label="Activation graph legend"><li><span class="high-line"></span>early · dashed</li><li><span class="middle-line"></span>middle · dotted</li><li><span class="low-line"></span>final · solid</li></ul>
    <p class="note">These views zoom into <b>blocks 1–6</b>, including a neighbor on each side of the tested 2–5 batch. The cosine axis is deliberately narrowed to <b>0.994–1.000</b>; its larger-looking differences are still small absolute changes. Relative L2 is <b>‖h_out − h_in‖₂ / ‖h_in‖₂</b>, reported as a percentage on a separate linear axis. Cosine compares direction; L2 also responds to changes in magnitude. The green wash identifies the tested batch, not an acceptance region.</p>
    <details class="activation-overview"><summary>Show all 32 blocks · full-range cosine and logarithmic relative L2</summary>
      <figure class="chart wide evidence-chart"><figcaption><b>Cosine · full range</b><i>vertical axis 0–1</i></figcaption><a href="activation-similarity.svg"><img src="activation-similarity.svg" width="820" height="390" alt="Full-range cosine across all 32 blocks at three stages, retaining the large changes at the first and final blocks."></a></figure>
      <figure class="chart wide evidence-chart"><figcaption><b>Relative L2 · all blocks</b><i>logarithmic axis · 5%–1,500%</i></figcaption><a href="activation-l2-all.svg"><img src="activation-l2-all.svg" width="820" height="390" alt="Relative L2 changes across all 32 blocks, on a logarithmic axis from 5% to 1,500%, so the large block-0 update does not flatten the remaining blocks."></a></figure>
      <p class="note">The L2 overview uses a log scale: equal vertical distances represent equal ratios rather than equal percentage-point changes. It preserves block 0’s update, which exceeds 800% of its input norm at the early stage, while keeping other blocks visible. All plotted observations fit the stated axes; none are clipped. The stage legend above applies to both overviews.</p>
    </details>
    {table(['Original block','Early cosine','Middle cosine','Final cosine'],activation_rows,'Mean input/output image-token cosine · 1 means same direction')}
    {table(['Original block','Early change','Middle change','Final change'],change_rows,'Relative hidden-state change · ‖h_out − h_in‖₂ / ‖h_in‖₂ · lower is smaller')}
    <p class="note">For example, block 2’s early cosine is 0.998321, but the update still has a norm equal to 8.681% of its input’s norm. High cosine does not imply a negligible update or 99.8% image quality. Block 1 and some later blocks also have high cosine; the shortlist was determined by the <b>maximum stage-average relative L2 change</b>, rather than cosine alone. These four earn a first test, not automatic approval for removal. Prompt-level distributions and per-capability aggregates are in <a href="results.json">the measurements</a>.</p>
    <h3>What happens when each update is removed?</h3><p class="summary">An identity bypass returns <code>h_out = h_in</code>, so the missing block contributes no update. We tested each candidate separately on the <b>same five pilot prompts × three stages</b>, using saved teacher-trajectory inputs and rebuilt student prefix caches. Each block has 15 velocity comparisons, with intact-teacher replay controls matching exactly within the diagnostic. These five-prompt Phase 3 probes are distinct from the later ten-prompt bridge-validation set.</p>
    <div class="charts">{bypass_charts}</div>
    {table(['Skipped block','Early error','Middle error','Final error','Worst case'],bypass_rows,'Phase 3 identity-bypass velocity error · five prompts · lower is better')}
    <p class="summary">Block 2 has the strongest activation similarity of the tested candidates early and in the middle, yet its omission causes the largest mean velocity errors at all three stages. Block 4 has the lowest mean bypass error early and in the middle; block 5 is lowest at the final stage. Similarity helps nominate experiments, but bypass sensitivity and finished images change the ranking.</p>
    <p class="note">The worst-case column is the largest error among that block’s 15 prompt/stage cases. Velocity error is a prediction difference, not a percentage of image-quality loss. These individual interventions do not establish that several blocks can be removed together, and no bridge is fitted in this diagnostic.</p>
    <h3>What the first image comparisons showed</h3><p class="summary">Five prompts, one per capability, were generated with block 4 or block 5 skipped: ten bypass images. In the astronaut example, skipping block 4 loses the spacesuit. Skipping block 5 retains it. That initially favors block 5 for closer study, while all four candidates remain eligible for learned bridges.</p>
    <p class="note">Each sheet reads <b>teacher / skip block 4 / skip block 5</b>, left to right. These are identity-bypass images from Phase 3, before bridge training or healing. Teacher images came from an earlier run; unresolved cross-run variation limits causal pixel comparisons. Observations are manual, not a blinded quality study. Open any sheet at full resolution.</p>
    {bypass_viewer}
    <p class="note">Pixel MAE and PSNR record changes in composition and appearance, but do not directly measure prompt adherence or aesthetic quality. None of these identity bypasses is accepted as the finished model.</p>
  </div></section>
  <section id="bridges" aria-labelledby="h-bridges"><div class="wrap"><p class="eyebrow">Phases 4–5 · recover the missing update cheaply</p><h2 id="h-bridges">A small residual bridge in place of a full block</h2>
    <div class="architecture" role="img" aria-label="Teacher: 32 transformer blocks. Student: first five blocks, rank-256 residual bridge replacing original block 5, then the 26 remaining blocks. Original slot numbers are zero-based."><div><span>Teacher</span><b>32 transformer blocks</b><small>Frozen reference</small></div><div><span>Student</span><b>Blocks 0–4 → bridge → blocks 6–31</b><small>31 original blocks + one learned replacement</small></div></div>
    <p class="summary">The bridge projects 4,096 features down to 256, applies GELU, projects back up, and adds the input: <code>h′ = h + W_up GELU(W_down h)</code>. Each bridge has 2,097,152 parameters. We fit four independent candidates, replacing one original block at a time; the four bridges are not inserted together.</p>
    <div class="charts bridge-diagrams">
      <figure class="chart"><figcaption><b>Before · the original transformer block</b></figcaption><a href="block-before.svg"><img src="block-before.svg" width="420" height="720" alt="Original block: 4,096-feature input, normalization, attention with a gated residual, normalization, feed-forward network with a gated residual, then a 4,096-feature output."></a></figure>
      <figure class="chart"><figcaption><b>After · a residual bottleneck bridge</b></figcaption><a href="bridge-after.svg"><img src="bridge-after.svg" width="420" height="720" alt="Replacement: preserve the input on a residual path; project 4,096 features down to 256, apply GELU, project back to a 4,096-feature correction, then add the original input."></a></figure>
    </div>
    <p class="note">The student keeps the <b>same 4,096-feature input and output interface</b>. Only the correction travels through the 256-feature bottleneck; the residual path keeps the original information. GELU is a nonlinear transformation of those 256 features. The bridge processes each token separately without attention and learns to approximate the removed block’s update. The diagrams show shapes and operations, not identical teacher and student values or measured speedups. Open either diagram for the standalone SVG.</p>
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
  <section id="healing" aria-labelledby="h-healing"><div class="wrap"><p class="eyebrow">Phase 6 | architecture healing</p><h2 id="h-healing">Full-student healing and generated-image QA</h2>
    <p class="summary">The original teacher has 32 transformer blocks. The current student has 31 surviving blocks and one pretrained bridge at slot 5. Phase 5 trained that bridge to replace the missing block's hidden-state update. Phase 6 then tested whether further training helps the student match the teacher's denoising velocity at the same prompt, noisy latent and timestep. Both models still use 40 denoising steps.</p>
    <p class="summary"><b>The intended healing pass updates the entire student:</b> all surviving transformer weights and the single bridge, 6,899,117,824 parameters in total. The teacher stays frozen. We attached no LoRA in the full-student runs. A separate LoRA for few-step speed distillation belongs to a later phase.</p>
    {table(['Experiment','Trainable weights','Updates','Final validation error','Selected update'],[
      ['Bridge + LoRA, rate 0.0001','10,223,616',120,'7.188%','0'],
      ['Bridge + LoRA, rate 0.00001','10,223,616',120,'6.113%','0'],
      ['Bridge only, rate 0.0001',f"{bridge_only['trainable_parameters']:,}",120,f"{100*bridge_only['evaluations'][-1]['mean_relative_l2']:.3f}%",bridge_only['selected_step']],
      ['Full student, rate 0.0001',f"{full_healing[0]['trainable_parameters']:,}",60,f"{100*full_healing[0]['evaluations'][-1]['mean_relative_l2']:.3f}%",full_healing[0]['selected_step']],
      ['Full student, rate 0.000001',f"{full_healing[1]['trainable_parameters']:,}",60,f"{100*full_healing[1]['evaluations'][-1]['mean_relative_l2']:.3f}%",full_healing[1]['selected_step']]
    ],'Phase 6 pilot results; mean velocity relative L2 on the same 10 validation prompts at three noise stages; lower is better')}
    <p class="note">All five pilots began at 5.518% with the pretrained bridge. Every saved training run selected update 0 on the original 10 prompts. The bridge + LoRA and bridge-only trials are diagnostic ablations; the two full-student trials answer the intended healing-scope question. A passed run means its execution controls passed, not that its trained checkpoint improved quality.</p>
    <h3>How the full-student trials worked</h3>
    <p class="summary">Each run made 60 shuffled updates from 20 training prompts at three saved noise stages. The loss combined teacher-student velocity error, hidden-state alignment at surviving slots 2-4, and bridge output alignment at slot 5. Ten different validation prompts chose among updates 0, 30 and 60. Exact teacher replay, baseline reproduction, gradient flow to the bridge and surviving weights, and selected-checkpoint reload were checked. Checkpoints, optimizer state, predictions, reports and hashes were saved separately.</p>
    {table(['Updates','Rate 0.0001','Rate 0.000001'],[[e['step'],f"{100*e['mean_relative_l2']:.3f}%",f"{100*full_healing[1]['evaluations'][i]['mean_relative_l2']:.3f}%"] for i,e in enumerate(full_healing[0]['evaluations'])],'Full-student mean validation velocity relative L2; 10 prompts times three stages')}
    <p class="summary">At update 60 the 0.0001 run reached 54.387%, a severe regression. Reducing the rate 100-fold kept the run near baseline, but its 5.654% still exceeded 5.518%. These short runs therefore did not produce a better healed student. They do not establish whether broader data, different loss weights or another optimizer could help.</p>
    <p class="summary">A closer look at the lower-rate run shows why its mean can rise despite local gains: <b>{validation_diagnostic['improved_comparisons']} of 30</b> validation prompt/stage comparisons improved, but the early-noise average rose from <b>{validation_diagnostic['by_stage'][0]['baseline_mean_percent']:.3f}% to {validation_diagnostic['by_stage'][0]['final_mean_percent']:.3f}%</b>. The largest regressions were early-noise cases. This is a small, reused validation set; the pattern guides diagnosis but does not identify a cause.</p>
    <h3>A broader check of the selected bridge</h3><p class="summary">We tested the unchanged pretrained-bridge student on 50 additional validation prompts, 10 from each prompt group, at early, middle and late noise stages. The original 10-prompt calibration measured <b>{100*expanded_validation['control_mean_relative_l2']:.3f}%</b> in this replay, close to the earlier 5.518%. The new 50-prompt mean was <b>{100*expanded_validation['new_mean_relative_l2']:.3f}%</b>. No student weight was updated.</p>
    {table(['Noise stage','Mean teacher-student velocity error'],[[r['stage'].title(),f"{100*r['mean_relative_l2']:.3f}%"] for r in expanded_validation['by_stage']],'Selected pretrained bridge on 50 new validation prompts; lower is closer to the teacher')}
    <p class="note">This extends the selected bridge's fixed-input check to 150 new prompt/stage comparisons. It is still a validation set, not an independent image-quality benchmark. Early noise remains the largest source of teacher-student disagreement.</p>
    <h3>The trained checkpoint on the same new inputs</h3><p class="summary">The low-rate update-60 checkpoint still regresses on the original 10 prompts: its control mean was <b>{100*expanded_trained['control_mean_relative_l2']:.3f}%</b>. On the 50 new prompts, however, it improved mean teacher agreement from <b>{expanded_comparison['pretrained_mean_percent']:.3f}% to {expanded_comparison['trained_mean_percent']:.3f}%</b>; {expanded_comparison['improved_comparisons']} of 150 paired prompt/stage comparisons improved. The new prompt selection was fixed before this checkpoint comparison. The disagreement between samples makes the original 10-prompt selection uncertain.</p>
    {table(['Noise stage','Pretrained bridge','Low-rate update 60'],[[r['stage'].title(),f"{r['pretrained_mean_percent']:.3f}%",f"{r['trained_mean_percent']:.3f}%"] for r in expanded_comparison['by_stage']],'Same 50 new prompts and saved teacher inputs; lower means closer velocity predictions')}
    <p class="note">This is teacher-prediction agreement on saved noisy inputs, not an image-quality result. The separate matched-image pilot below examines what these numerical gains look like after full generation.</p>
    <h3>Longer full-student healing</h3>
    <p class="summary">We resumed the low-rate full-student checkpoint at update 60 with its saved optimizer state. Forty previously unused training prompts provided 120 new prompt/noise-stage updates; no validation prompt entered training. All 6,899,117,824 student weights were trainable, with one bridge and no LoRA. On the 50 monitored validation prompts, mean velocity relative L2 fell from <b>{100*continuation['evaluations'][0]['mean_relative_l2']:.3f}% at update 60</b> to <b>{100*continuation['evaluations'][1]['mean_relative_l2']:.3f}% at update 120</b> and <b>{100*continuation['evaluations'][2]['mean_relative_l2']:.3f}% at update 180</b>.</p>
    <p class="summary">On 20 separate prompts untouched by training and checkpoint monitoring, the means were <b>{100*continuation_test['means']['pretrained']:.3f}% pretrained</b>, <b>{100*continuation_test['means']['update60']:.3f}% at update 60</b>, <b>{100*continuation_test['means']['update120']:.3f}% at update 120</b>, and <b>{100*continuation_test['means']['update180']:.3f}% at update 180</b>. Update 120 was best on this small test; the extra updates to 180 did not improve it further there.</p>
    <p class="note">These are teacher-prediction differences on saved noisy inputs. Early-noise agreement worsened with more updates while late-noise agreement improved; the overall test gain is small. The image comparison below tests full generations separately.</p>
    <h3>What the generated images show</h3>
    <p class="summary">The first pilot generated 10 fixed-seed teacher/pretrained-bridge image pairs. A later pilot generated <b>11 matched triples</b> with teacher, pretrained bridge, and low-rate update 60. The two student versions usually retain similar subjects and styles; their composition and detail differences are mixed. In the poster below, the teacher renders the requested CRIMSON HARBOR title while both students replace HARBOR with unrelated letters. The trained checkpoint does not fix this known text failure.</p>
    <div class="paired-images"><figure class="comparison"><a href="images/phase6-poster-teacher.png"><img loading="lazy" src="images/phase6-poster-teacher.png" alt="Teacher poster with the requested CRIMSON HARBOR title"></a><figcaption><b>Original teacher</b> | requested title</figcaption></figure><figure class="comparison"><a href="images/phase6-poster-pretrained.png"><img loading="lazy" src="images/phase6-poster-pretrained.png" alt="Pretrained-bridge poster with the second title word incorrect"></a><figcaption><b>Pretrained bridge</b> | title error</figcaption></figure><figure class="comparison"><a href="images/phase6-poster-trained.png"><img loading="lazy" src="images/phase6-poster-trained.png" alt="Update-60 poster with the second title word still incorrect"></a><figcaption><b>Low-rate update 60</b> | title error remains</figcaption></figure></div>
    <p class="note">This visual review is small and unblinded. It does not establish a quality ranking: a lower teacher-velocity error on saved inputs has not shown a clear generated-image advantage here. Full-resolution PNGs are available by opening each poster.</p>
    <h3>Five-way image comparison after more updates</h3>
    <p class="summary">Eleven fixed-prompt, fixed-seed cases now compare the teacher, pretrained bridge, and cumulative updates 60, 120, and 180. Only <b>two prompts request prominent text</b>. The Northern Parade title is legible in every student version. In the other poster, <b>update 120 renders CRIMSON HARBOR and Visit Kyoto legibly</b>; the pretrained and update-60 versions miss HARBOR, and update 180 reads SIMSON and omits HARBOR. That is <b>not a steady text improvement or deterioration</b> as healing proceeds. Replacing block 5 with the bridge harms this particular poster relative to the teacher, but two posters cannot establish general bridge text fidelity. Most other image differences are subtle or mixed.</p>
    <div class="paired-images"><figure class="comparison"><a href="images/phase6-continuation-poster-teacher.png"><img loading="lazy" src="images/phase6-continuation-poster-teacher.png" alt="Teacher CRIMSON HARBOR poster"></a><figcaption><b>Teacher</b></figcaption></figure><figure class="comparison"><a href="images/phase6-continuation-poster-pretrained.png"><img loading="lazy" src="images/phase6-continuation-poster-pretrained.png" alt="Pretrained bridge poster"></a><figcaption><b>Pretrained bridge</b></figcaption></figure><figure class="comparison"><a href="images/phase6-continuation-poster-update60.png"><img loading="lazy" src="images/phase6-continuation-poster-update60.png" alt="Update 60 poster"></a><figcaption><b>Update 60</b></figcaption></figure><figure class="comparison"><a href="images/phase6-continuation-poster-update120.png"><img loading="lazy" src="images/phase6-continuation-poster-update120.png" alt="Update 120 poster with legible CRIMSON HARBOR and Visit Kyoto"></a><figcaption><b>Update 120</b> | title legible</figcaption></figure><figure class="comparison"><a href="images/phase6-continuation-poster-update180.png"><img loading="lazy" src="images/phase6-continuation-poster-update180.png" alt="Update 180 poster with title regression"></a><figcaption><b>Update 180</b> | title regresses</figcaption></figure></div>
    <p class="note">The viewer below shows all 11 prompts: four on the first sheet, four on the second, and three on the third. Each row holds the same prompt and seed across the five columns: teacher, pretrained, update 60, update 120, update 180. Open a sheet for full resolution.</p>
    {five_way_viewer}
    <p class="note">This inspection was unblinded and includes one previously tracked poster. Only two of these 11 prompts request prominent text. The fresh text-focused test below checks whether that poster result repeats.</p>
    <h3>A fresh test of the update-120 text hypothesis</h3>
    <p class="summary">We rendered <b>16 previously unused explicit-text prompts</b>: eight posters and eight other text scenes. Each uses the same fixed prompt and seed for teacher, pretrained bridge, update 120, and update 180, producing 64 saved images. The rubric was written before image review: a prompt passes only when every requested phrase is readable; ambiguous lettering remains uncertain. No weights were updated.</p>
    {table(['Model','Clear full-text passes','Uncertain'],[[role, text_review['counts'][role].get('true',0),text_review['counts'][role].get('uncertain',0)] for role in ('teacher','pretrained','update120','update180')],'Manual full-requested-text review on 16 fixed prompts; each prompt counts once')}
    <p class="summary">Update 120 <b>clearly restores the Northern Orchard title</b> that the pretrained bridge and update 180 omit, although its small Visit Kyoto tagline is borderline. On a fresh <b>ENDLESS HARBOR / Visit Kyoto</b> poster, all three students render both phrases correctly, so the CRIMSON HARBOR failure did not repeat on that close template. Update 180 breaks <b>Midnight River</b>, which pretrained and update 120 render correctly. The strict full-prompt score is tied between pretrained and update 120 at 13 clear passes; update 120 has one additional uncertain case. This supports a local recovery and an update-180 regression, not a general claim that update 120 fixes text.</p>
    <p class="note">The viewer below shows all 16 prompts, four per sheet, with the four models in each row. The <a href="images/phase6-text-p1-9225ba4ccf76-update120.png">ENDLESS HARBOR</a>, <a href="images/phase6-text-p1-5962716b5dc8-update120.png">Northern Orchard</a>, and <a href="images/phase6-text-p1-30b3b241cfaa-update120.png">Midnight River</a> update-120 PNGs can also be opened at full resolution; each sheet links to its full-size image. The review is unblinded and uses one seed per prompt.</p>
    {text_pilot_viewer}
    <p class="note">The observed 40-step generation averages on these 10 ordered pairs were {public['paired_images']['teacher_mean_seconds']:.2f} seconds for the teacher and {public['paired_images']['student_mean_seconds']:.2f} seconds for the student, a {public['paired_images']['teacher_mean_seconds']/public['paired_images']['student_mean_seconds']:.2f}x ratio. This ordered pilot does not control for run order or establish a production speed benchmark. Velocity error measures denoising predictions, not image quality.</p>
    <h3>All 45 remaining validation text prompts</h3>
    <p class="summary">To check whether update 120 consistently improves text, we rendered all <b>45 other unused validation prompts with explicit text</b>. The teacher, pretrained bridge, update 120, and update 180 produced 180 matched 40-step images at one fixed seed per prompt. A written rubric required every requested phrase to be readable. One AI visual reviewer scored anonymous A/B/C/D columns before revealing model identities; uncertain text was recorded separately. This step trained no weights.</p>
    {table(['Model','Clear full-text passes','Clear failures','Uncertain'],[[role, text_qa_review['counts'][role].get('pass',0),text_qa_review['counts'][role].get('fail',0),text_qa_review['counts'][role].get('uncertain',0)] for role in ('teacher','pretrained','update120','update180')],'45 matched prompts; each prompt counts once and passes only if all requested phrases are readable')}
    <p class="summary">The pretrained bridge and update 120 received <b>the same status on every prompt</b>: 41 clear passes, three failures, and one uncertain. Update 120 clearly beat update 180 on two posters, PAPER STATION and NORTHERN PARADE. The teacher had 43 clear passes. This broader qualitative check gives no general text-rendering advantage to update 120 over pretrained, despite its CRIMSON HARBOR recovery in the earlier image pilot.</p>
    <p class="note">All four models garbled tiny Moonlight Motel print in one storefront and missed the Live in Berlin tagline on GOLDEN RIVER. Three student images spelled Fresh Bread Daily with an extra letter. The Electric Frontier flag lettering was too small to judge confidently in three student images. Literal text scoring also misses layout mistakes: one image has Live Music Tonight at the top but garbled lettering where the prompt requested it in sand. The viewer below contains all 15 labeled sheets, three prompts per sheet. One seed and one AI reviewer do not establish a population-level quality ranking; the separate internal test prompts remain unused.</p>
    {text_qa_viewer}
    <p class="note">These 45 prompts mainly ask for exact words on signs, posters, menus, or other surfaces. Turning each into a visual question would mostly repeat the text score. The broader 1,002-prompt validation split also includes 260 alignment prompts with counts, colors, objects, and spatial relations that support more specific question-based checks. Aesthetics and fine-detail prompts need different evidence; one question score cannot rank overall image quality.</p>
    <h3 id="checkpoint-decision">Deciding test: 129 blind prompt-adherence questions</h3>
    <p class="summary">Velocity agreement and text tests had not clearly separated update 120 from the pretrained bridge, so one last test asked whether update 120 <b>follows concrete prompt requirements more often</b>. We used <b>every unused validation prompt</b> in four checkable categories: counting, spatial relations, colors, and multiple objects. That is 129 prompts in English and Chinese. Each was rendered by both models with the same seed, size, and 40-step sampler: <b>258 matched images</b> and no training. Each category had one fixed question, such as <i>"Are the requested number(s) of each named object visible?"</i> Images were scored pass, fail, or uncertain as anonymous A/B columns before the model key was opened. The promotion rule was saved before any image existed.</p>
    {decision_tiles}
    {decision_svg}
    {rule_table}
    <div class="verdict"><p><b>Decision: keep the pretrained bridge.</b> Update 120 won 3 prompts, lost none, and tied on 126. That is a +{gain:.1f}-point gain whose 95% interval reaches down to 0.0. It misses both promotion conditions. The healing run was safe, causing no measured regression, but its benefit is too small to detect in every Phase 6 test. The pretrained bridge also remains the simpler artifact: only the bridge is trained, and the 31 surviving blocks keep the teacher's original weights.</p></div>
    {pass_chart}
    {group_table}
    <h3>What the differences look like</h3>
    <p class="summary">Below are <b>all five prompts where the two models scored differently</b>, followed by two typical failures shared by both and one typical tie. The pretrained bridge is always on the left and update 120 on the right; tags show the blind scores. The three clear wins are real: an occluded giraffe, the right count of octopuses, and a missing candle restored. Against 129 prompts, however, they are too few to clear the predeclared bar. Most prompts look like the last example: almost the same image, same verdict.</p>
    <div class="decision-examples">{example_figures}</div>
    <p class="note">Both checkpoints still fail about half of the counting prompts and about a third of the spatial-relation prompts in the same way. Those errors come from the base model and the bridge, not from healing. The viewer below shows all 129 prompts unblinded, three per sheet, with the score for every image.</p>
    {align_viewer}
    <h3>Research decision: all Phase 6 evidence side by side</h3>
    {evidence_table}
    <p class="summary">Every comparison points the same way. Update 120 is <b>numerically a little closer to the teacher</b> on saved inputs: lower velocity error on 46 of 60 untouched comparisons, by 0.04 points. It ties on text and is <b>not reliably better on generated images</b>. Phase 6 therefore closes with the <b>pretrained single-bridge student as the selected model</b>. The full-student checkpoints remain saved as evidence, but none is promoted.</p>
    <p class="note">Limits: one seed per prompt and one AI visual reviewer. A real effect of a few points cannot be ruled out, but this test was sized to detect the 5-point gain needed to justify the extra healing training. Per-prompt scores, the rubric, the blind key, and hashes are in the downloadable record.</p>
    <h3 id="next-step">Then: remove all four blocks at once</h3>
    <p class="summary">Phase 6 removed <b>one</b> block, and the pretrained bridge alone was enough. We did not assume that holds for more blocks, so <a href="#phase7">Phase 7</a> removes all four candidate blocks (2, 3, 4 and 5) at once and asks the same question: <b>do we just use the bridges, or do we need healing?</b> Healing is now conditional rather than a fixed stage.</p>
    <p class="note">The LoRA in this plan is for <b>speed</b>, not healing. Once the 28-block architecture is fixed, a speed-distillation LoRA will teach it to follow the teacher's 40-step trajectories in about 8 larger steps. That is roughly 5x fewer transformer passes, against about 1.14x from removing four blocks.</p>
  </div></section>
  <section id="phase7" aria-labelledby="h-phase7"><div class="wrap"><p class="eyebrow">Phase 7 | how many blocks can go?</p><h2 id="h-phase7">Four blocks breaks the image. Two blocks looks like the teacher</h2>
    <p class="summary">Removing one block (Phase 6) is too small a cut to be worth releasing. Phase 7 tried bigger cuts. We first removed <b>all four candidate blocks, 2, 3, 4 and 5</b>, going from 32 blocks to 28, then tried a smaller cut of <b>only blocks 4 and 5</b>. Each student is compared with the teacher on the same noisy inputs. Its <b>velocity error</b> is how different its denoising answer is from the teacher's, where 0% means identical. The reference is the student with only block 5 removed. The predeclared gate allows at most 1 point more than that student on 10 check prompts, which is 6.52%.</p>
    <h3>Four blocks, bridges only: not enough</h3>
    <p class="summary">We tried the cheap options first: the four Phase 5 bridges stacked, and one bridge for the whole removed region at two sizes. All 90 teacher predictions were matched exactly first.</p>
    {p7_chart}
    {p7_table}
    <p class="summary">Blocks 2 and 3 are the costly ones. Even removed alone, their own bridges recover little, unlike blocks 4 and 5:</p>
    {p7_block_table}
    <p class="summary">The region bridge learns the text-prompt tokens almost perfectly, but leaves about 12% of the image hidden state unexplained. A bridge transforms each token on its own, while the removed blocks mix all tokens through attention, so quadrupling its size barely helps:</p>
    {p7_fit_table}
    <h3>Four blocks, healed: better, but not close</h3>
    <p class="summary">Healing trains all 6.25 billion surviving weights and the bridge against the teacher. The first attempt gave each training step one prompt at one noise moment. The <b>early moment</b>, the first denoising step where the layout is decided, has about ten times larger errors and gradients, and it broke. Giving every step all three moments with equal weight fixed that at small learning rates, and 3e-6 was selected from a four-rate grid. Further passes then leveled off. On the 60 training prompts the early moment improved by about 60%, but on new prompts by only 4%: with just 60 saved early inputs, the student memorized them. We captured <b>{len(p7_capture['examples'])} extra early-moment examples</b>, each verified by two identical deterministic teacher passes. They stopped the memorizing, but the early moment stayed unstable.</p>
    {p7_path_chart}
    {p7_heal_table}
    <div class="verdict"><p><b>After 480 training steps the 28-block student reaches {100*p7_pool_eval[420]['mean_relative_l2']:.1f}% on the 50 wider prompts, against {100*expanded_validation['new_mean_relative_l2']:.1f}% for the one-block student.</b> Progress had slowed to a crawl, and most of the gap is at the early moment.</p></div>
    <h3>Two blocks: much closer</h3>
    <p class="summary">So we tested the smaller cut, removing only blocks 4 and 5, which were the easy ones on their own. One bridge maps the teacher's state before block 4 to its state after block 5. It was fitted with the same recipe, then measured in the full 30-block student:</p>
    {p7_45_table}
    <p class="summary">The best bridge, with 4.2M parameters, reaches <b>{100*p7_45['region_rank512', 'all']:.2f}%</b>, 2.45 points above the one-block student. Healing it with the four-block recipe improves the middle and late moments steadily, but the early moment gets worse after step 60:</p>
    {p7_heal45_table}
    <div class="verdict"><p><b>No 30-block student passes the 6.52% gate.</b> The selected healed checkpoint, step {p7_heal45['selected_step']}, measures {100*p7_heal45_eval[60]['mean_relative_l2']:.2f}% on the wider prompts and {100*p7_heal45_eval[60]['gate_mean_relative_l2']:.2f}% on the check prompts.</p></div>
    <h3 id="phase7-images">What the images show</h3>
    <p class="summary">The gate measures answer agreement, not pictures. So we generated the same 30 validation prompts with the teacher and four candidates: 25 prompts never used before, plus 5 text prompts. Every model got the same prompt, seed, size and 40 steps. A, B and C are the 30-block students, healed at steps 60 and 240 and bridge only. D is the healed 28-block student.</p>
    {p7_candidates_viewer}
    {p7_text_table}
    <p class="note">Failures: {p7_text_failures}.</p>
    <div class="verdict"><p><b>The 28-block student is not usable.</b> It draws wrong subjects, including a cat for a dog, an eye for a peacock feather and a harbour for a biplane, and it fails every text prompt. <b>The 30-block students look like the teacher.</b> Sharpness, lighting, anatomy and style match. They often draw a different but valid composition, as another seed would. The healed step-60 student (A) rendered all five text prompts correctly, like the teacher, and is the strongest candidate.</p></div>
    <p class="note">The image review is a first screen by one unblinded reviewer (Claude), on 30 prompts with one seed each. It is not a blinded benchmark. Removing two of 32 blocks is a modest cut, and its actual speed gain has not been measured. A larger blinded comparison of candidate A against the teacher, and a search for further removable blocks, are under consideration.</p>
  </div></section>
  <section id="record" aria-labelledby="h-record"><div class="wrap"><p class="eyebrow">The research record</p><h2 id="h-record">What is complete, what remains open</h2>
    <dl class="facts"><div><dt>Phase 0</dt><dd>Architecture inspected: 32 blocks and 4,096-wide hidden states.</dd></div><div><dt>Phase 1</dt><dd>20k prompt pilot serialized, with a 100k template pool and fixed split assignments.</dd></div><div><dt>Phase 2</dt><dd>Original-model images, judged baseline, repeatability analysis and H100 efficiency measurements saved.</dd></div><div><dt>Phase 3</dt><dd>Layer diagnostics and identity-bypass image pilots; blocks 2–5 retained as learned-replacement candidates.</dd></div><div><dt>Phase 4</dt><dd>Residual bottleneck bridges implemented and insertion checked.</dd></div><div><dt>Phase 5</dt><dd>Four independent bridge-pretraining pilots completed; checkpoints and optimizer states saved.</dd></div><div><dt>Phase 6</dt><dd>Repeatable teacher contract verified; bridge and full-student healing pilots saved. The 120-update continuation was compared on an untouched 20-prompt fixed-input test, an 11-prompt five-way comparison, 16- and 45-prompt text reviews, and a blinded 129-prompt adherence test. <b>Complete: the pretrained bridge is selected</b>; update 120 (86 vs 83 clear passes, +2.3 points, 95% CI [0.0, 5.4]) missed the predeclared promotion bar.</dd></div><div><dt>Phase 7</dt><dd>In progress. Removing blocks 2-5 (28 blocks) failed even after healing: {100*p7_pool_eval[420]['mean_relative_l2']:.1f}% velocity error against {100*expanded_validation['new_mean_relative_l2']:.1f}% for the one-block student, with broken images. Removing only blocks 4-5 (30 blocks) reaches {100*p7_heal45_eval[60]['mean_relative_l2']:.2f}% after healing. It is still above the numerical gate, but in a first 30-prompt image screen it <b>looks like the teacher</b>, and the healed step-60 student rendered all 5 text prompts correctly.</dd></div><div><dt>Later phases</dt><dd>A blinded comparison of the 30-block candidate against the teacher, a search for further removable blocks, a speed-distillation LoRA from 40 to about 8 steps, speed measurements and optional quantization remain planned. No results are claimed for them.</dd></div></dl>
    <h3>Reproducibility and updates</h3><p class="summary">Each executable notebook step has a method description and a plain-language explanation of its result. Targets, raw outputs, metrics, selected weights, optimizer state and manifests are serialized locally, with expensive experiment artifacts backed up remotely. Matching saved steps reuse their results; changed inputs stop rather than silently overwrite the record.</p>
    <ul class="get"><li><a class="primary" href="results.json">Download the measurements<small>JSON · exact values and source SHA-256 hashes</small></a></li><li><a href="https://github.com/josejuanmartinez/nihonga">Research repository ↗<small>Phase notebooks and original plan</small></a></li><li><a href="https://github.com/Neopolita/niwaki-page/tree/main/nihonga">Page source ↗<small>Public research record</small></a></li><li><a href="healing.svg">Earlier bridge + LoRA graph<small>Standalone SVG · diagnostic ablation</small></a></li></ul>
    <p class="note">This page is a dated snapshot of saved experiments, not a live training dashboard. Its refresh script reads selected local summaries and copies existing comparison images; it runs no models and publishes no weights, caches or credentials. Every plotted value is available in the downloadable record.</p>
  </div></section>
  <section id="status" aria-labelledby="h-status"><div class="wrap">
    <p class="eyebrow">Current result · development pilot</p><h2 id="h-status">Two blocks removed looks like the teacher. Four blocks breaks the image</h2>
    <p class="summary">Phase 6 showed that one removed block, 5, needs only its pretrained bridge: <b>5.518%</b> velocity error, with healing adding nothing measurable. That cut is too small to release. In <a href="#phase7">Phase 7</a>, removing four blocks (2-5) stayed far from the teacher even after 480 healing steps: <b>{100*p7_pool_eval[420]['mean_relative_l2']:.1f}%</b> against <b>{100*expanded_validation['new_mean_relative_l2']:.1f}%</b>, with wrong subjects and broken text in the images. Removing only blocks 4 and 5, with one 4.2M-parameter bridge and 60 healing steps, reaches <b>{100*p7_heal45_eval[60]['mean_relative_l2']:.2f}%</b>. That is still above the predeclared gate, but in a first <a href="#phase7-images">30-prompt image screen</a> it looks like the teacher and renders all five text prompts correctly.</p>
    <ul class="figures"><li><span class="n">30 <small>+ 1</small></span><span class="c">surviving transformer blocks plus one bridge in the strongest candidate, from 32 original blocks</span></li><li><span class="n">{100*p7_heal45_eval[60]['mean_relative_l2']:.2f}<small>%</small></span><span class="c">velocity error on 50 validation prompts, against {100*expanded_validation['new_mean_relative_l2']:.2f}% for the one-block student</span></li><li><span class="n">5 <small>/ 5</small></span><span class="c">text prompts rendered correctly, like the teacher</span></li><li><span class="n">0 <small>/ 5</small></span><span class="c">text prompts correct for the healed 28-block student</span></li></ul>
    <p class="note">Velocity errors are measured on saved teacher inputs at three denoising moments, not on finished images. The image screen is unblinded, uses one seed per prompt, and is not a benchmark. Removing 2 of 32 blocks is a modest cut, and its speed gain has not been measured.</p>
  </div></section>
 </main>
 <script src="text-qa-viewer.js" defer></script>
'''
(out/'index.html').write_text(head+body+footer,encoding='utf-8')
print('Wrote Nihonga documentation, measurement snapshot, vector graph, galleries, the Phase 6 decision evidence and Phase 7 results.')
