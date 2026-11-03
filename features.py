"""Deterministic features; no references, question IDs, or learned lookup tables."""
import ast
from fractions import Fraction
import math
import re
import numpy as np
from common import parse_answer

NUMBER = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
WORDS = re.compile(r'[a-z]+')
EQUATION = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?(?:\s*[+*/-]\s*-?\d+(?:\.\d+)?){1,8})\s*=\s*(-?\d+(?:\.\d+)?)')
STYLE = ['therefore', 'however', 'wait', 'assum', 'alternativ', 'check', 'let ', 'total', 'answer', 'correct', 'approximately', 'round', 'step', 'but ', 'instead', 'mistake']


def arithmetic(text):
    # Numeric grammar parsed by AST. No eval/exec or variable substitution.
    t = text.replace(',', '').replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/').replace('×', '*').replace('÷', '/').replace('−', '-')
    def calc(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float): return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return calc(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = calc(n.left), calc(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            return a/b
        raise ValueError('Unsupported arithmetic')
    good = bad = 0
    for expression, rhs in EQUATION.findall(t):
        try:
            value = calc(ast.parse(expression.strip(), mode='eval').body)
            if math.isclose(value, float(rhs), rel_tol=1e-3, abs_tol=0.011): good += 1
            else: bad += 1
        except (ValueError, ZeroDivisionError, SyntaxError, OverflowError): pass
    return [good, bad, good/(1+good+bad), bad/(1+good+bad)]


def extract_features(bank):
    rows, groups = [], []
    names = None
    for row in bank:
        question = row['question'].lower(); qw = set(WORDS.findall(question)); qnums = set(NUMBER.findall(question))
        values = [parse_answer(c['text']) for c in row['candidates']]; groups.append(values)
        valid = [v for v in values if v is not None]; unique = len(set(valid)); local = []
        parsed_qnums = {parse_answer('#### '+n) for n in qnums}
        for c, value in zip(row['candidates'], values):
            text = c['text']; lower = text.lower(); tokens = WORDS.findall(lower); tw = set(tokens); nums = NUMBER.findall(text)
            try: answer = float(Fraction(value)) if value is not None else 0.0
            except (ValueError, OverflowError): answer = 0.0
            answer = float(np.clip(answer, -1e15, 1e15)); count = values.count(value) if value is not None else 0
            other = [float(Fraction(v)) for v in valid]; median = float(np.median(other)) if other else 0.0
            d = {
                'log_chars': math.log1p(len(text)), 'log_words': math.log1p(len(tokens)),
                'lines': math.log1p(text.count('\n')), 'digits': math.log1p(sum(ch.isdigit() for ch in text)),
                'numbers': math.log1p(len(nums)), 'equals': math.log1p(text.count('=')),
                'latex': math.log1p(text.count('\\')), 'bold': math.log1p(text.count('**')),
                'question_marks': text.count('?'), 'boxed': text.count('\\boxed'), 'hash_answer': int('####' in text),
                'question_overlap': len(tw & qw)/max(1,len(qw)), 'question_number_overlap': len(set(nums)&qnums)/max(1,len(qnums)),
                'lexical_diversity': len(tw)/max(1,len(tokens)),
                'valid_answer': int(value is not None), 'answer_log_abs': math.log1p(abs(answer)),
                'answer_negative': int(answer < 0), 'answer_zero': int(value is not None and answer == 0),
                'answer_integer': int(value is not None and Fraction(value).denominator == 1),
                'answer_in_question': int(value is not None and value in parsed_qnums),
                'answer_mentions': sum(parse_answer('#### '+n)==value for n in nums) if value is not None else 0,
                'answer_support': count/4, 'valid_count': len(valid)/4, 'distinct_answers': unique/4,
                'majority': int(count == max([values.count(v) for v in valid], default=0)),
                'answer_log_median_delta': math.copysign(math.log1p(abs(answer-median)),answer-median),
                'question_chars': math.log1p(len(question)), 'question_numbers': len(qnums),
            }
            for word in STYLE: d['style_'+word.strip()] = lower.count(word)
            for name, val in zip(['arithmetic_good','arithmetic_bad','arithmetic_good_fraction','arithmetic_bad_fraction'], arithmetic(text)): d[name] = val
            if names is None: names = list(d)
            local.append(list(d.values()))
        a = np.asarray(local, dtype=np.float64)
        rows.append(np.concatenate([a, a-a.mean(axis=0,keepdims=True)], axis=1))
    return np.asarray(rows), names+[n+'_relative' for n in names] if names else [], groups
