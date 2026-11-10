import pytest
from practice.store import Store
from practice.grading import grade
from practice.importers import parse_text_document, validate_bundle


def question(**changes):
    q=dict(id='test',source_id='Q1',chapter='1',type='single',stem='哪项正确？',options=[{'id':'a','text':'错误'},{'id':'b','text':'正确'}],correct=['b'],answers=[],explanation='说明',evidence='证据',reviewed=True,version=1)
    q.update(changes)
    return q


def test_stable_ids_and_exact_cloze():
    q=question(options=[{'id':'b','text':'正确'},{'id':'a','text':'错误'}])
    assert grade(q,['b'])['correct']
    assert not grade(q,['a'])['correct']
    q=question(type='cloze',options=[],correct=[],answers=[['马尔可夫','Markov']])
    assert grade(q,[' ＭＡＲＫＯＶ '])['correct']
    assert not grade(q,['不是马尔可夫'])['correct']
    assert not grade(q,[])['correct']


def test_multiple_and_numeric_units():
    q=question(type='multiple',correct=['a','b'])
    assert grade(q,['b','a'])['correct']
    assert not grade(q,['a'])['correct']
    q=question(type='cloze',options=[],correct=[],answers=[['48']],numeric={'value':48,'tolerance':0.01,'unit':'个百分点'})
    assert grade(q,['48个百分点'])['correct']
    assert not grade(q,['48%'])['correct']
    assert not grade(q,['148'])['correct']


def test_versions_are_durable_and_attempts_not_rewritten(tmp_path):
    p=tmp_path/'bank.sqlite';s=Store(p)
    s.import_bundle({'sources':[{'id':'Q1','number':1,'title':'标题','answer':'证据','chapter':'1'}], 'questions':[question()]})
    s.record_attempt('test',1,['b'])
    s.update_question('test',question(correct=['a']))
    s=Store(p)
    assert s.attempts()[0]['correct'] is True
    assert s.attempts()[0]['version']==1
    assert s.get_question('test')['version']==2
    assert s.stats()['attempted']==0 # current revision still unattempted
    with pytest.raises(ValueError):s.record_attempt('test',1,['b'])


def test_invalid_bundle_is_atomic_and_duplicate_import(tmp_path):
    s=Store(tmp_path/'db')
    with pytest.raises(ValueError):s.import_bundle({'sources':[], 'questions':[question()]})
    assert not s.questions()
    bundle={'sources':[{'id':'Q1','number':1,'title':'标题','answer':'证据','chapter':'1'}],'questions':[question()]}
    assert s.import_bundle(bundle)['added_questions']==1
    assert s.import_bundle(bundle)['duplicate'] is True
    assert len(s.questions())==1
    exported=s.export_bundle();s2=Store(tmp_path/'copy');s2.import_bundle(exported)
    assert s2.questions()==s.questions()


def test_import_preserves_question_and_answer():
    b=parse_text_document('# 第一部分\n\n## Q2. 注意力\n\n原始答案。\n\n## Q3. 多头\n\n另一答案。','test.md')
    assert [x['number'] for x in b['sources']]==[2,3]
    assert b['sources'][0]['answer']=='原始答案。'
    assert b['questions']==[] # no fake auto-generated objective questions
    with pytest.raises(ValueError):validate_bundle({'sources':b['sources'],'questions':[question(source_id='missing')]})


def test_restore_is_atomic_and_preserves_all_revisions(tmp_path):
    s=Store(tmp_path/'db');s.import_bundle({'sources':[{'id':'Q1','number':1,'title':'标题','answer':'证据','chapter':'1'}],'questions':[question()]})
    s.record_attempt('test',1,['b']);s.update_question('test',question(correct=['a']))
    backup=s.export_bundle();s.restore(backup)
    assert s.get_question('test')['version']==2
    assert s.attempts()[0]['correct']
    bad={**backup,'attempts':[{'id':'broken','question_id':'nope','version':9,'correct':True}]}
    with pytest.raises(ValueError):s.restore(bad)
    assert s.export_bundle()==backup


def test_upstream_retrieval_handles_chinese():
    from app.retrieval.bm25 import BM25Index
    from app.corpus.models import Document
    index=BM25Index.build([Document('one','a','标题','机器人动作连续性'),Document('two','b','标题','统计置信区间')])
    assert index.search('动作')[0][0].doc_id=='one'
