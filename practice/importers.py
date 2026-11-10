"""Use upstream parsers, then preserve source QAs without inventing answers."""
import hashlib
import json
import math
import re
from app.ingestion.parsers import parse_file


def validate_question(q, source_ids):
    if not isinstance(q,dict):raise ValueError('题目必须是对象')
    q=dict(q)
    for key in ['id','source_id','chapter','stem','type']:
        if not isinstance(q.get(key),str) or not q[key].strip():raise ValueError(f'题目缺少 {key}')
    if len(q['id'])>180 or len(q['stem'])>20000:raise ValueError('题目过长')
    if q['source_id'] not in source_ids:raise ValueError('题目引用的原文不存在')
    if q['type'] not in ['single','multiple','cloze']:raise ValueError('不支持的题型')
    q.setdefault('version',1);q.setdefault('reviewed',False)
    if not isinstance(q['version'],int) or q['version']<1:raise ValueError('版本号不正确')
    q.setdefault('options',[]);q.setdefault('correct',[]);q.setdefault('answers',[])
    if q['type']=='cloze':
        if not isinstance(q['answers'],list) or not q['answers'] or not all(isinstance(a,list) and a and all(isinstance(x,str) and x.strip() for x in a) for a in q['answers']):raise ValueError('填空题必须有每个空的标准答案')
        if '____' not in q['stem']:raise ValueError('填空位置需使用 ____ 标记')
        if len(re.findall(r'_{4,}',q['stem'])) != len(q['answers']):raise ValueError('填空位置和答案数量不一致')
        if q.get('numeric'):
            rule=q['numeric']
            if len(q['answers'])!=1 or not all(isinstance(rule.get(k), (float,int)) and math.isfinite(rule[k]) for k in ['value','tolerance']) or rule['tolerance']<0:raise ValueError('数值题的值或容差不正确')
    else:
        opts=q['options']
        if not isinstance(opts,list) or not 2<=len(opts)<=8 or not all(isinstance(o,dict) and isinstance(o.get('id'),str) and isinstance(o.get('text'),str) and o['text'].strip() for o in opts):raise ValueError('选择题需有 2 至 8 个有效选项')
        ids=[o['id'] for o in opts]
        if len(ids)!=len(set(ids)):raise ValueError('选项 ID 重复')
        if not isinstance(q['correct'],list) or not q['correct'] or not set(q['correct']).issubset(ids) or len(set(q['correct']))!=len(q['correct']):raise ValueError('正确选项无效')
        if q['type']=='single' and len(q['correct'])!=1:raise ValueError('单选题必须只有一个正确选项')
    return q


def validate_bundle(bundle, existing_sources=()):
    if not isinstance(bundle,dict) or not isinstance(bundle.get('sources'),list) or not isinstance(bundle.get('questions'),list):raise ValueError('需要包含 sources 和 questions 数组的题库 JSON')
    if len(bundle['sources'])>2000 or len(bundle['questions'])>5000:raise ValueError('单次导入题目太多')
    ids=set(existing_sources)
    new=[]
    for s in bundle['sources']:
        if not isinstance(s,dict) or not all(isinstance(s.get(k),str) and s[k].strip() for k in ['id','title','answer','chapter']):raise ValueError('原文需要 id/title/answer/chapter')
        if s['id'] in new:raise ValueError('原文 ID 重复')
        new.append(s['id']);ids.add(s['id'])
    questions=[validate_question(q,ids) for q in bundle['questions']]
    qids=[q['id'] for q in questions]
    if len(qids)!=len(set(qids)):raise ValueError('题目 ID 重复')
    return {**bundle,'questions':questions}


def parse_text_document(text, filename):
    prefix=hashlib.sha256(text.encode()).hexdigest()[:12]
    pattern=re.compile(r'(?m)^(?:#{1,4}\s+)?(?:\*\*)?Q(\d+)[.．、\s]+([^\n]+)')
    matches=list(pattern.finditer(text));sources=[]
    for i,m in enumerate(matches):
        answer=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
        answer=re.split(r'\n# (?:面试前|资料与|常用术语)',answer)[0].strip()
        title=m[2].strip().rstrip('*').strip()
        # A chapter heading immediately before the next Q belongs to that next Q.
        answer=re.sub(r'\n(?:#{1,4}[^\n]+\n?)+\s*$','',answer).strip()
        if not answer:continue
        chapter_match=list(re.finditer(r'(?m)^#{1,3}\s+(第[^\n]+)',text[:m.start()]))
        chapter=chapter_match[-1][1] if chapter_match else '导入题库'
        sources.append(dict(id=f'import-{prefix}-Q{m[1]}',number=int(m[1]),title=title,chapter=chapter,chapter_title=chapter,answer=answer,images=[],citations=[],filename=filename))
    if not sources:
        if not text.strip():raise ValueError('没有可读取的正文；扫描 PDF 可能需要先做 OCR')
        sources=[dict(id=f'import-{prefix}',number=0,title=filename,chapter='导入资料',chapter_title='导入资料',answer=text.strip(),images=[],citations=[],filename=filename)]
    return {'format':'vla-bank-v1','sources':sources,'questions':[]}


def read_import(path, filename):
    if path.suffix.lower()=='.json':
        try:bundle=json.loads(path.read_text('utf-8'))
        except (ValueError,UnicodeError) as e:raise ValueError('JSON 格式或编码不正确') from e
        return validate_bundle(bundle)
    text=parse_file(path)
    if len(text)>2_000_000:raise ValueError('正文过长，请拆分文件')
    return parse_text_document(text,filename)
