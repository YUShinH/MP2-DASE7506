"""Conservative numeric consistency checks, not a proof of word-problem semantics.

Never evaluates arbitrary Python, resolves variables, or reads gold answers.
Unsupported/ambiguous equations are counted separately from contradictions.
"""
import ast
from fractions import Fraction
import math
import re
import numpy as np
from common import parse_answer

FRAC=re.compile(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}')
SPAN=re.compile(r'[\d.\s()+*/%=~^\-]+')
NAMES=['checked','exact','rounded','wrong','undefined','unknown','coverage','wrong_fraction','rounded_fraction','has_wrong','has_undefined','last_matches_answer','last_differs_answer','answer_comparable','final_markers','conflicting_markers','fraction_syntax','percent_syntax','parenthesis_syntax','chain_links']


def normalize(text):
    t=text.replace('−','-').replace('–','-').replace('×','*').replace('÷','/').replace('\\%','%')
    t=re.sub(r'^\s*[-*]\s+(?=\$|\*\*|\\\()','',t,flags=re.M)
    t=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',t)
    for _ in range(12):
        changed=FRAC.sub(lambda m:'(('+m[1]+')/('+m[2]+'))',t)
        if changed==t: break
        t=changed
    t=re.sub(r'\\(?:times|cdot)\b','*',t); t=re.sub(r'\\div\b','/',t)
    t=re.sub(r'\\(?:approx|simeq)\b','~',t).replace('≈','~')
    t=t.replace('\\left','').replace('\\right','').replace('\\!','').replace('\\,',' ')
    t=re.sub(r'\\text\{\s*of\s*\}',' of ',t)
    t=re.sub(r'\\boxed\{([^{}]*)\}',r'(\1)',t)
    t=re.sub(r'\\(?:text|mathrm|operatorname)\{[^{}]*\}','|',t)
    t=re.sub(r'\*\*([^*\n]+)\*\*',r'\1',t)  # Markdown bold, before exponent conversion.
    t=t.replace('$',' ').replace('\\(',' ').replace('\\)',' ').replace('\\[',' ').replace('\\]',' ')
    t=re.sub(r'(?<![\w.])(-?)(\d+)\s+(\d+)\s*/\s*(\d+)',r'\1(\2+\3/\4)',t)
    t=re.sub(r'%\s+of\b','% *',t,flags=re.I)
    t=re.sub(r'(?<![\w.])(\d+(?:\.\d+)?)\s*%',r'(\1/100)',t)
    t=re.sub(r'\)\s+of\s+(?=[\d(])',') * ',t,flags=re.I)
    t=re.sub(r'\b(?:one\s+)?half\s+of\s+(?=[\d(])','(1/2) * ',t,flags=re.I)
    t=re.sub(r'\b(?:one\s+)?quarter\s+of\s+(?=[\d(])','(1/4) * ',t,flags=re.I)
    t=re.sub(r'\btwice\s+(?=[\d(])','2 * ',t,flags=re.I)
    for word,operator in [('plus','+'),('minus','-'),('times','*'),('divided by','/')]:
        t=re.sub(r'(?<=\d)\s+'+word+r'\s+(?=[\d(])',' '+operator+' ',t,flags=re.I)
    t=re.sub(r'(?<=\d)\s*(?=\()','*',t); t=re.sub(r'\)\s*(?=[\d(])',')*',t)
    t=t.replace('^{','^(').replace('}',')') if '^{' in t else t
    return t


def numeric(expression):
    """Bounded AST arithmetic. Fraction retains exact rational calculations."""
    expression=expression.strip().replace('^','**')
    if not expression or len(expression)>512: raise ValueError('expression length')
    node=ast.parse(expression,mode='eval')
    if len(list(ast.walk(node)))>100: raise ValueError('expression complexity')
    def visit(n):
        if isinstance(n,ast.Expression): return visit(n.body)
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            if isinstance(n.value,float) and not math.isfinite(n.value): raise ValueError('nonfinite')
            value=Fraction(str(n.value))
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            value=visit(n.operand)*(-1 if isinstance(n.op,ast.USub) else 1)
        elif isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add): value=a+b
            elif isinstance(n.op,ast.Sub): value=a-b
            elif isinstance(n.op,ast.Mult): value=a*b
            elif isinstance(n.op,ast.Div): value=a/b
            else:
                if b.denominator!=1 or abs(b)>12: raise ValueError('exponent bound')
                value=a**int(b)
        else: raise ValueError('unsupported syntax')
        if value.numerator.bit_length()>256 or value.denominator.bit_length()>256 or abs(value)>10**18: raise ValueError('value bound')
        return value
    return visit(node)


def check_pair(left,right,approx=False):
    try: a,b=numeric(left),numeric(right)
    except ZeroDivisionError: return 'undefined',None,None
    except (SyntaxError,ValueError,TypeError,OverflowError,RecursionError): return 'unknown',None,None
    if a==b: return 'exact',a,b
    # Percent precision is measured in percentage points, including rounded displays.
    percent=re.fullmatch(r'\((\d+(?:\.(\d+))?)/100\)',right.strip())
    if percent:
        places=len(percent[2] or ''); tolerance=Fraction(1,2*10**places)
        if re.search(r'\*\s*100\s*\)*$',left):
            if a==b*100: return 'exact',a,b*100
            if abs(a-b*100)<=tolerance: return 'rounded',a,b*100
        elif abs(a-b)<=tolerance/100: return 'rounded',a,b
    # Explicit decimal rounding or an approximation sign; never silently allow integer rounding at '='.
    decimal=re.fullmatch(r'[+-]?\d+\.(\d+)',right.strip())
    if decimal or approx:
        places=len(decimal[1]) if decimal else 0
        tolerance=Fraction(1,2*10**places)
        if abs(a-b)<=tolerance: return 'rounded',a,b
    return 'wrong',a,b


def inspect(text):
    normalized=normalize(text); checks=[]; separators=len(re.findall(r'[=~]',normalized)); last=None
    for line in normalized.splitlines():
        for match in SPAN.finditer(line):
            chunk=match.group().strip().rstrip('.')
            if '=' not in chunk and '~' not in chunk: continue
            parts=re.split(r'([=~])',chunk)
            for j in range(0,len(parts)-2,2):
                left,right=parts[j].strip(),parts[j+2].strip().rstrip('.')
                # A leading operator after excluded text may be a suffix of an unknown expression.
                prefix=line[:match.start()]
                ambiguous=(j==0 and left.startswith(('+','-','*','/','^')) and match.start()>0 and prefix.strip()[-1:] not in ('',':'))
                ambiguous=ambiguous or (j==0 and bool(prefix) and prefix[-1].isalpha() and match.group().startswith('('))
                ambiguous=ambiguous or (j==len(parts)-3 and match.end()<len(line) and line[match.end()].isalpha() and not match.group()[-1].isspace())
                ambiguous=ambiguous or (j==0 and bool(re.fullmatch(r'[+-]?\d+(?:\.\d+)?',left)) and bool(re.search(r'[A-Za-z]\s*$',prefix)))
                ambiguous=ambiguous or (j==0 and match.start()==0 and bool(re.match(r'^\s*-\s+[\d(]',match.group())))
                # A trailing arithmetic operator before a variable must not be truncated.
                if ambiguous: state,a,b='unknown',None,None
                else: state,a,b=check_pair(left,right,parts[j+1]=='~')
                checks.append(dict(left=left,right=right,status=state))
                if a is not None and b is not None: last=b
    counts={s:sum(c['status']==s for c in checks) for s in ['exact','rounded','wrong','undefined','unknown']}
    counts['unknown']=max(counts['unknown'],separators-sum(counts[s] for s in ['exact','rounded','wrong','undefined']))
    checked=sum(counts[s] for s in ['exact','rounded','wrong'])
    official=parse_answer(text); answer=Fraction(official) if official is not None else None
    markers=[]
    for value in re.findall(r'\\boxed\{([^{}]+)\}',text):
        try: markers.append(numeric(normalize(value)))
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): pass
    for value in re.findall(r'####[^\n]*',text):
        v=parse_answer(value)
        if v is not None: markers.append(Fraction(v))
    comparable=answer is not None and last is not None
    values=[math.log1p(checked),math.log1p(counts['exact']),math.log1p(counts['rounded']),math.log1p(counts['wrong']),math.log1p(counts['undefined']),math.log1p(counts['unknown']),checked/max(1,separators),counts['wrong']/max(1,checked),counts['rounded']/max(1,checked),int(counts['wrong']>0),int(counts['undefined']>0),int(comparable and last==answer),int(comparable and last!=answer),int(comparable),math.log1p(len(markers)),int(len(set(markers))>1),math.log1p(len(FRAC.findall(text))),math.log1p(text.count('%')),math.log1p(text.count('(')),math.log1p(sum(max(0,len(re.findall('[=~]',m.group()))-1) for m in SPAN.finditer(normalized)))]
    return dict(counts=counts,checked=checked,separators=separators,last_result=str(last) if last is not None else None,official_answer=official,last_differs_answer=bool(comparable and last!=answer),conflicting_markers=len(set(markers))>1,checks=checks,features=values)


def extract_math_features(bank):
    if not bank: return np.zeros((0,4,2*len(NAMES)),dtype=np.float32)
    raw=np.asarray([[inspect(c['text'])['features'] for c in row['candidates']] for row in bank],dtype=np.float32)
    return np.concatenate([raw,raw-raw.mean(1,keepdims=True)],axis=-1)
