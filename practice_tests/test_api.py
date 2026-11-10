import io
from fastapi.testclient import TestClient
from practice.main import create_app
from practice_tests.test_bank import question

HEAD={'X-VLA-Request':'1'}

def client(tmp_path):
    app=create_app(tmp_path,seed=False,testing=True)
    app.state.store.import_bundle({'sources':[{'id':'Q1','number':1,'title':'标题','answer':'证据','chapter':'1'}],'questions':[question()]})
    return TestClient(app)


def test_bootstrap_grading_and_csrf(tmp_path):
    c=client(tmp_path);boot=c.get('/api/bootstrap').json()
    assert len(boot['questions'])==1
    assert 'correct' not in boot['questions'][0]
    assert c.post('/api/attempts',json={'question_id':'test','version':1,'answer':['b']}).status_code==403
    assert c.post('/api/attempts',headers={**HEAD,'Origin':'https://evil.example'},json={}).status_code==403
    r=c.post('/api/attempts',headers=HEAD,json={'question_id':'test','version':1,'answer':['b']})
    assert r.json()['correct'] is True
    assert c.get('/api/bootstrap').json()['stats']['correct']==1
    assert c.get('/api/bootstrap',headers={'Host':'evil.example'}).status_code==403


def test_import_preview_confirm_roundtrip(tmp_path):
    c=client(tmp_path)
    r=c.post('/api/import/preview',headers=HEAD,files={'file':('new.md','## Q2. 标题\n\n答案'.encode())})
    assert r.status_code==200,r.text
    preview=r.json();assert len(preview['sources'])==1
    assert len(c.get('/api/bootstrap').json()['sources'])==1
    a=c.post('/api/import/confirm',headers=HEAD,json={'token':preview['token']});assert a.json()['added_sources']==1
    assert c.post('/api/import/confirm',headers=HEAD,json={'token':'../anything'}).status_code==400
    backup=c.get('/api/export').content
    assert c.post('/api/restore',headers=HEAD,files={'file':('backup.json',backup)}).status_code==200
    assert len(c.get('/api/bootstrap').json()['sources'])==2


def test_missing_ai_does_not_fake_generation(tmp_path,monkeypatch):
    monkeypatch.delenv('VLA_AI_MODEL',raising=False)
    c=client(tmp_path)
    r=c.post('/api/generate',headers=HEAD,json={'source_ids':['Q1'],'type':'single','count':1})
    assert r.status_code==503
    assert c.get('/api/bootstrap').json()['review_drafts']==[]

def test_backup_restores_review_drafts(tmp_path):
    c=client(tmp_path)
    draft={**question(),'id':'draft-1','reviewed':False}
    c.app.state.store.add_review_drafts([draft])
    backup=c.get('/api/export').content
    assert c.post('/api/review/draft-1',headers=HEAD,json={'action':'reject'}).status_code==200
    assert c.post('/api/restore',headers=HEAD,files={'file':('backup.json',backup)}).status_code==200
    assert c.get('/api/bootstrap').json()['review_drafts']==[draft]

def test_same_material_renamed_is_duplicate(tmp_path):
    c=client(tmp_path)
    for filename in ('a.txt','b.txt'):
        r=c.post('/api/import/preview',headers=HEAD,files={'file':(filename,'重复导入的正文'.encode())})
        assert r.status_code==200
        r=c.post('/api/import/confirm',headers=HEAD,json={'token':r.json()['token']})
        assert r.status_code==200
        if filename=='b.txt':assert r.json()['duplicate'] is True
