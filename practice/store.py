"""Versioned SQLite bank and durable personal practice records."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid
from practice.grading import grade
from practice.importers import validate_bundle,validate_question


def now():return datetime.now(timezone.utc).isoformat()
def loads(row):return json.loads(row['data'])


class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.db() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS questions(id TEXT,version INTEGER,data TEXT NOT NULL,PRIMARY KEY(id,version));
            CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS favorites(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS code_drafts(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS code_runs(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS review_drafts(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            ''')
    @contextmanager
    def db(self):
        c=sqlite3.connect(self.path,timeout=10);c.row_factory=sqlite3.Row
        try:
            with c:yield c
        finally:c.close()
    def _rows(self,table):
        with self.db() as c:return [loads(r) for r in c.execute(f'SELECT data FROM {table} ORDER BY rowid')]
    def sources(self):return self._rows('sources')
    def questions(self,public=False):
        with self.db() as c:
            qs=[loads(r) for r in c.execute('SELECT q.data FROM questions q JOIN (SELECT id,MAX(version) v FROM questions GROUP BY id) m ON q.id=m.id AND q.version=m.v ORDER BY q.rowid')]
        return [self.public(q) if public else q for q in qs if q.get('reviewed')]
    @staticmethod
    def public(q):return {**{k:v for k,v in q.items() if k not in ['correct','answers','numeric','explanation','evidence']},'blank_count':len(q.get('answers',[])) if q['type']=='cloze' else 0}
    def get_question(self,id):
        with self.db() as c:r=c.execute('SELECT data FROM questions WHERE id=? ORDER BY version DESC LIMIT 1',(id,)).fetchone()
        if not r:raise ValueError('题目不存在')
        return loads(r)
    def import_bundle(self,bundle):
        bundle=validate_bundle(bundle,[s['id'] for s in self.sources()]);ns=nq=0
        with self.db() as c:
            for s in bundle['sources']:
                old=c.execute('SELECT data FROM sources WHERE id=?',(s['id'],)).fetchone()
                if old:
                    previous=loads(old)
                    def content_only(source):
                        result=dict(source);filename=result.pop('filename',None)
                        if filename and result.get('title')==filename:result['title']=''
                        return result
                    if content_only(previous)!=content_only(s):raise ValueError(f"原文 ID 冲突：{s['id']}，请更换 ID 或导入完整备份")
                if not old:c.execute('INSERT INTO sources VALUES(?,?)',(s['id'],json.dumps(s,ensure_ascii=False)));ns+=1
            for q in bundle['questions']:
                old=c.execute('SELECT data FROM questions WHERE id=? ORDER BY version DESC LIMIT 1',(q['id'],)).fetchone()
                if old:
                    if loads(old)!=q:raise ValueError(f"题目 ID 冲突：{q['id']}，请通过编辑保存新版本")
                    continue
                c.execute('INSERT INTO questions VALUES(?,?,?)',(q['id'],q['version'],json.dumps(q,ensure_ascii=False)));nq+=1
        return {'added_sources':ns,'added_questions':nq,'duplicate':ns==0 and nq==0}
    def update_question(self,id,q):
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE')
            old=c.execute('SELECT data FROM questions WHERE id=? ORDER BY version DESC LIMIT 1',(id,)).fetchone()
            if not old:raise ValueError('题目不存在')
            old=loads(old)
            q=validate_question({**q,'id':id,'version':old['version']+1,'reviewed':True},[r[0] for r in c.execute('SELECT id FROM sources')])
            c.execute('INSERT INTO questions VALUES(?,?,?)',(id,q['version'],json.dumps(q,ensure_ascii=False)))
        return q
    def record_attempt(self,id,version,answer):
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute('SELECT data FROM questions WHERE id=? ORDER BY version DESC LIMIT 1',(id,)).fetchone()
            if not row:raise ValueError('题目不存在')
            q=loads(row)
            if version!=q['version']:raise ValueError('题目已更新，请刷新后重新作答')
            if not isinstance(answer,list) or len(answer)>20 or not all(isinstance(v,str) and len(v)<2000 for v in answer):raise ValueError('答案格式不正确')
            result=grade(q,answer);a=dict(id=uuid.uuid4().hex,question_id=id,version=version,answer=answer,correct=result['correct'],created_at=now())
            c.execute('INSERT INTO attempts VALUES(?,?)',(a['id'],json.dumps(a,ensure_ascii=False)))
        return {**result,'attempt':a}
    def attempts(self):return self._rows('attempts')
    def favorites(self):
        with self.db() as c:return [r[0] for r in c.execute('SELECT id FROM favorites')]
    def favorite(self,id,active):
        self.get_question(id)
        with self.db() as c:
            if active:c.execute('INSERT OR IGNORE INTO favorites VALUES(?)',(id,))
            else:c.execute('DELETE FROM favorites WHERE id=?',(id,))
    def save_draft(self,id,code):
        with self.db() as c:c.execute('INSERT OR REPLACE INTO code_drafts VALUES(?,?)',(id,json.dumps({'id':id,'code':code,'updated_at':now()},ensure_ascii=False)))
    def code_drafts(self):return {d['id']:d['code'] for d in self._rows('code_drafts')}
    def record_code_run(self,id,result):
        r=dict(id=uuid.uuid4().hex,exercise_id=id,created_at=now(),**result)
        with self.db() as c:c.execute('INSERT INTO code_runs VALUES(?,?)',(r['id'],json.dumps(r,ensure_ascii=False)))
        return r
    def review_drafts(self):return self._rows('review_drafts')
    def add_review_drafts(self,questions):
        with self.db() as c:
            for q in questions:c.execute('INSERT INTO review_drafts VALUES(?,?)',(q['id'],json.dumps(q,ensure_ascii=False)))
    def review(self,id,action,q=None):
        if action not in ['approve','reject']:raise ValueError('审核操作无效')
        with self.db() as c:
            row=c.execute('SELECT data FROM review_drafts WHERE id=?',(id,)).fetchone()
            if not row:raise ValueError('草稿不存在或已审核')
            if action=='approve':
                item={**(q or loads(row)),'id':id,'reviewed':True,'version':1}
                item=validate_question(item,[r[0] for r in c.execute('SELECT id FROM sources')])
                c.execute('INSERT INTO questions VALUES(?,?,?)',(id,1,json.dumps(item,ensure_ascii=False)))
            c.execute('DELETE FROM review_drafts WHERE id=?',(id,))
    def stats(self):
        qs=self.questions();latest={}
        for a in self.attempts():latest[(a['question_id'],a['version'])]=a
        chapters={}
        for q in qs:
            ch=chapters.setdefault(q['chapter'],dict(chapter=q['chapter'],total=0,attempted=0,correct=0));ch['total']+=1
            a=latest.get((q['id'],q['version']))
            if a:ch['attempted']+=1;ch['correct']+=int(a['correct'])
        attempted=sum(x['attempted'] for x in chapters.values());correct=sum(x['correct'] for x in chapters.values())
        return dict(total=len(qs),attempted=attempted,correct=correct,wrong=attempted-correct,accuracy=round(100*correct/attempted,1) if attempted else 0,by_chapter=list(chapters.values()),code_runs=self._rows('code_runs'))
    def clear_progress(self):
        with self.db() as c:
            for t in ['attempts','favorites','code_drafts','code_runs']:c.execute(f'DELETE FROM {t}')
    def export_bundle(self):
        return dict(format='vla-bank-v1',sources=self.sources(),questions=self.questions(),question_versions=self._rows('questions'),attempts=self.attempts(),favorites=self.favorites(),code_drafts=self.code_drafts(),code_runs=self._rows('code_runs'),review_drafts=self.review_drafts())
    def restore(self,bundle):
        bundle=validate_bundle(bundle)
        versions=bundle.get('question_versions',bundle['questions'])
        ids={s['id'] for s in bundle['sources']}
        versions=[validate_question(q,ids) for q in versions]
        review_drafts=bundle.get('review_drafts',[])
        if not isinstance(review_drafts,list):raise ValueError('审核草稿格式错误')
        review_drafts=[validate_question(q,ids) for q in review_drafts]
        draft_ids=[q['id'] for q in review_drafts]
        if len(set(draft_ids))!=len(draft_ids) or any(q.get('reviewed') for q in review_drafts):raise ValueError('审核草稿重复或状态无效')
        pairs={(q['id'],q['version']) for q in versions}
        if len(pairs)!=len(versions):raise ValueError('备份版本重复')
        if not {(q['id'],q['version']) for q in bundle['questions']}.issubset(pairs):raise ValueError('备份题目版本缺失')
        for t in ['attempts','code_runs']:
            rows=bundle.get(t,[])
            if not isinstance(rows,list) or not all(isinstance(x,dict) and isinstance(x.get('id'),str) for x in rows):raise ValueError('备份记录格式错误')
        for a in bundle.get('attempts',[]):
            if (a.get('question_id'),a.get('version')) not in pairs or not isinstance(a.get('correct'),bool):raise ValueError('作答记录引用了无效版本')
        favorites=bundle.get('favorites',[]);drafts=bundle.get('code_drafts',{})
        if not isinstance(favorites,list) or not all(isinstance(v,str) for v in favorites):raise ValueError('收藏格式错误')
        if not isinstance(drafts,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in drafts.items()):raise ValueError('代码草稿格式错误')
        with self.db() as c:
            for t in ['sources','questions','attempts','favorites','code_drafts','code_runs','review_drafts']:c.execute(f'DELETE FROM {t}')
            for row in bundle['sources']:c.execute('INSERT INTO sources VALUES(?,?)',(row['id'],json.dumps(row,ensure_ascii=False)))
            for row in versions:c.execute('INSERT INTO questions VALUES(?,?,?)',(row['id'],row['version'],json.dumps(row,ensure_ascii=False)))
            for t in ['attempts','code_runs']:
                for row in bundle.get(t,[]):c.execute(f'INSERT INTO {t} VALUES(?,?)',(row['id'],json.dumps(row,ensure_ascii=False)))
            for id in favorites:c.execute('INSERT INTO favorites VALUES(?)',(id,))
            for id,code in drafts.items():c.execute('INSERT INTO code_drafts VALUES(?,?)',(id,json.dumps({'id':id,'code':code,'updated_at':now()})))
            for row in review_drafts:c.execute('INSERT INTO review_drafts VALUES(?,?)',(row['id'],json.dumps(row,ensure_ascii=False)))
