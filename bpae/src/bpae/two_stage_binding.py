"""BPAE v4: training-global anomaly presence followed by deterministic binding.

Old evidence_policy.select_verified_candidate is deliberately unchanged.
This module accepts only question/options, measured values and frozen train rules.
"""
from __future__ import annotations
import math,re
from .binding import map_text_to_features,_numbers,_claimed_direction

NO_ANOMALY=re.compile(r'^No selected acoustic feature shows an unusual deviation under (?:the|these) reference rules\.?$',re.I)

def active_rules(values,rules):
    active=[]
    for r in rules:
        if r.get('partition')!='19LA_train' or r.get('group')!='global':
            raise ValueError('Only 19LA_train/global rules are allowed')
        f=r['feature'];v=values.get(f)
        if v is None or not math.isfinite(float(v)):continue
        v=float(v);t=r['rule_type']
        if t=='low':hit=v<float(r['upper_threshold']);direction='low'
        elif t=='high':hit=v>float(r['lower_threshold']);direction='high'
        elif t=='two':
            lo,hi=float(r['lower_threshold']),float(r['upper_threshold'])
            hit=v<lo or v>hi;direction='low' if v<lo else 'high'
        else:raise ValueError('Unknown frozen rule')
        if hit:active.append({'feature':f,'value':v,'direction':direction,'ranking_score':float(r['raw_balanced_accuracy'])})
    return sorted(active,key=lambda r:(-r['ranking_score'],r['feature']))

def two_stage_binding(question,values,rules):
    opts=question['options'];neutral=[k for k,v in opts.items() if NO_ANOMALY.fullmatch(v.strip())]
    if len(neutral)!=1:raise ValueError('Exactly one canonical no-anomaly option required')
    active=active_rules(values,rules);by={r['feature']:r for r in active}
    result={'version':'two_stage_binding_v4','anomaly_present':bool(active),'active_rules':active,'candidate_checks':[]}
    if not active:return dict(result,prediction=neutral[0],reason='no_frozen_rule_triggered')
    ranked=[]
    for letter,text in sorted(opts.items()):
        if letter==neutral[0]:continue
        direction=_claimed_direction(text)
        for f in map_text_to_features(text):
            item=by.get(f)
            if item is None:continue
            for claimed,decimals in _numbers(text):
                error=abs(item['value']-claimed);tol=0.5*10**(-decimals)+1e-9
                check={'option':letter,'feature':f,'numeric_error':error,'normalized_error':error/(abs(claimed)+tol),'numeric_match':error<=tol,'direction_match':direction==item['direction'],'claimed_direction':direction,'measured_direction':item['direction']}
                result['candidate_checks'].append(check)
                if check['numeric_match'] and check['direction_match']:
                    ranked.append((check['normalized_error'],-item['ranking_score'],f,letter))
    # Presence is decided before candidate inspection. Do not turn an unmatched
    # anomaly into a no-anomaly answer; abstain when no option can be verified.
    return dict(result,prediction=min(ranked)[-1] if ranked else None,reason='verified_feature_value_direction' if ranked else 'anomaly_present_but_no_candidate_verified')

def bpae_t2_v4(question,values,rules):
    """Public BPAE binding entry point: ALM cannot overwrite this decision."""
    return two_stage_binding(question,values,rules)
