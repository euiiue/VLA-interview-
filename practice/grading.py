"""Deterministic, deliberately conservative objective grading."""
import math
import re
import unicodedata


def normalize(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value)).casefold())


def grade(q, answer):
    expected = q.get('correct', []) if q['type'] != 'cloze' else q.get('answers', [])
    correct = False
    if isinstance(answer, list) and all(isinstance(a, str) for a in answer):
        if q['type'] in ('single', 'multiple'):
            correct = len(answer) == len(set(answer)) and set(answer) == set(expected)
        elif len(answer) == len(expected):
            correct = all(normalize(v) in {normalize(a) for a in aliases} for v, aliases in zip(answer, expected))
            if q.get('numeric') and len(answer) == 1:
                rule = q['numeric'];value = normalize(answer[0]);unit=normalize(rule.get('unit',''))
                if unit and value.endswith(unit):value=value[:-len(unit)]
                try:
                    correct = bool(re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',value)) and math.isfinite(float(value)) and abs(float(value)-rule['value']) <= rule.get('tolerance',0)
                except (ValueError,OverflowError):correct=False
    return {'correct':bool(correct),'expected':expected,'explanation':q.get('explanation','')}
