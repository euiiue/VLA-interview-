"""Loopback-only service for practice, assets and static React UI."""
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
import zipfile
from fastapi import FastAPI,Request,UploadFile,File,Body,HTTPException
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from practice.store import Store
from practice.runner import DockerRunner
from practice.importers import read_import
from practice.generation import ai_environment,generate_drafts

ROOT=Path(__file__).resolve().parents[1]
from dotenv import load_dotenv
load_dotenv(ROOT/'.env.practice',override=False)
DATA=Path(__file__).resolve().parent/'data'


def exercises(private=False):
    from practice.exercises.catalog import get_exercises
    return get_exercises(include_private=private)


def create_app(data_dir=None,seed=True,testing=False,runner=None):
    app=FastAPI(title='VLA 面试练习室',docs_url=None,redoc_url=None)
    local=Path(data_dir or os.getenv('VLA_DATA_DIR',str(ROOT/'.local')));local.mkdir(parents=True,exist_ok=True)
    store=Store(local/'practice.sqlite');app.state.store=store
    engine=runner or DockerRunner();previews=local/'previews';previews.mkdir(exist_ok=True)
    assets=ROOT/'practice/assets';assets.mkdir(exist_ok=True)
    env_cache={}
    def seed_once():
        if seed and not store.sources() and (DATA/'sources.json').exists() and (DATA/'questions.json').exists():
            store.import_bundle({'sources':json.loads((DATA/'sources.json').read_text()),'questions':json.loads((DATA/'questions.json').read_text())})
    seed_once()
    @app.middleware('http')
    async def local_boundary(req:Request,call_next):
        host=req.url.hostname
        if host not in (['127.0.0.1','localhost','testserver'] if testing else ['127.0.0.1','localhost']):return JSONResponse({'detail':'只允许本机访问'},403)
        if req.method not in ['GET','HEAD','OPTIONS']:
            origin=req.headers.get('origin')
            if origin:
                parsed=urlparse(origin)
                allowed=parsed.hostname in ['127.0.0.1','localhost'] and parsed.scheme=='http' and parsed.port in [8765,3000,3001]
                if testing and origin=='http://testserver':allowed=True
                if not allowed:return JSONResponse({'detail':'跨站请求被拒绝'},403)
            if req.headers.get('x-vla-request')!='1':return JSONResponse({'detail':'缺少同源请求标记'},403)
        response=await call_next(req)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        return response
    @app.exception_handler(ValueError)
    async def invalid(req,exc):return JSONResponse({'detail':str(exc)},400)
    @app.exception_handler(RuntimeError)
    async def unavailable(req,exc):return JSONResponse({'detail':str(exc)},503)
    @app.get('/api/bootstrap')
    def bootstrap():
        seed_once()
        if time.monotonic()-env_cache.get('at',-100)>5:env_cache.update(at=time.monotonic(),value=engine.environment())
        sources=store.sources();chapters=[]
        for s in sources:
            if not any(c['id']==s['chapter'] for c in chapters):chapters.append({'id':s['chapter'],'title':s.get('chapter_title',s['chapter'])})
        try:code=exercises()
        except ImportError:code=[]
        return dict(sources=sources,questions=store.questions(True),exercises=code,stats=store.stats(),environment={**env_cache['value'],**ai_environment()},chapters=chapters,favorites=store.favorites(),attempts=store.attempts(),code_drafts=store.code_drafts(),review_drafts=store.review_drafts())
    @app.get('/api/questions/{id}/edit')
    def edit(id:str):return store.get_question(id)
    @app.patch('/api/questions/{id}')
    def update(id:str,body:dict=Body(...)):return {'question':store.update_question(id,body.get('question',{}))}
    @app.post('/api/attempts')
    def attempt(body:dict=Body(...)):
        return store.record_attempt(body.get('question_id'),body.get('version'),body.get('answer'))
    @app.post('/api/favorites')
    def favorite(body:dict=Body(...)):
        store.favorite(body.get('question_id'),body.get('active') is True);return {'ok':True}
    def get_exercise(id,private=False):
        item=next((x for x in exercises(private) if x['id']==id),None)
        if not item:raise ValueError('代码题不存在')
        return item
    @app.get('/api/exercises/{id}/reference')
    def reference(id:str):return {'reference':get_exercise(id,True)['reference']}
    @app.post('/api/code/draft')
    def code_draft(body:dict=Body(...)):
        id=body.get('exercise_id');get_exercise(id)
        code=body.get('code')
        if not isinstance(code,str) or len(code)>100000:raise ValueError('代码为空或超过 100 KB')
        store.save_draft(id,code);return {'ok':True}
    @app.post('/api/code/run')
    def code_run(body:dict=Body(...)):
        id=body.get('exercise_id');get_exercise(id)
        code=body.get('code')
        if not isinstance(code,str) or len(code)>100000:raise ValueError('代码格式不正确或超过 100 KB')
        store.save_draft(id,code);result=engine.run(id,code);store.record_code_run(id,result)
        return result
    async def upload(file):
        data=await file.read(15_000_001)
        if len(data)>15_000_000:raise ValueError('文件超过 15 MB，请拆分')
        suffix=Path(file.filename or '').suffix.lower()
        if suffix not in ['.json','.md','.txt','.pdf','.docx']:raise ValueError('支持 JSON、MD、TXT、PDF、DOCX')
        return data,suffix,Path(file.filename or '资料').name
    @app.post('/api/import/preview')
    async def preview(file:UploadFile=File(...)):
        data,suffix,name=await upload(file)
        with tempfile.TemporaryDirectory(prefix='vla-import-') as temp:
            path=Path(temp)/('upload'+suffix);path.write_bytes(data)
            try:
                if suffix=='.docx':
                    with zipfile.ZipFile(path) as z:
                        if sum(x.file_size for x in z.infolist())>50_000_000:raise ValueError('DOCX 解压后过大')
                bundle=read_import(path,name)
            except ValueError:raise
            except Exception as e:raise ValueError('文件无法读取，请检查格式或先转为 Markdown/TXT') from e
        token=uuid.uuid4().hex
        for old in previews.glob('*.json'):
            if time.time()-old.stat().st_mtime>1800:old.unlink(missing_ok=True)
        if len(list(previews.glob('*.json')))>50:raise ValueError('待确认导入过多，请先完成或稍后重试')
        (previews/f'{token}.json').write_text(json.dumps(bundle,ensure_ascii=False))
        existing={x['id'] for x in store.sources()};duplicate=bool(bundle['sources']) and all(x['id'] in existing for x in bundle['sources'])
        warnings=[]
        if not bundle['questions']:warnings.append('已识别为来源材料，未自动编造选择题。入库后可配置 AI 生成草稿，或导入结构化题库 JSON。')
        if suffix in ['.pdf','.docx']:warnings.append('公式、表格和图片不能仅凭文本提取保证完整，请对照原文复核。')
        return dict(token=token,filename=name,duplicate=duplicate,sources=bundle['sources'],questions=bundle['questions'],warnings=warnings)
    @app.post('/api/import/confirm')
    def confirm(body:dict=Body(...)):
        token=body.get('token','')
        if not isinstance(token,str) or not re.fullmatch('[a-f0-9]{32}',token):raise ValueError('预览凭证无效')
        path=previews/f'{token}.json'
        if not path.exists() or time.time()-path.stat().st_mtime>1800:raise ValueError('预览已过期，请重新导入')
        result=store.import_bundle(json.loads(path.read_text()));path.unlink();return result
    @app.post('/api/generate')
    def generate(body:dict=Body(...)):
        ids=body.get('source_ids',[])
        if not isinstance(ids,list) or not all(isinstance(i,str) for i in ids):raise ValueError('原文 ID 无效')
        source_map={s['id']:s for s in store.sources()}
        if not ids or any(i not in source_map for i in ids):raise ValueError('请选择有效原文')
        count=body.get('count',1)
        if not isinstance(count,int):raise ValueError('题目数量必须是整数')
        result=generate_drafts([source_map[i] for i in ids],body.get('type','single'),count)
        store.add_review_drafts(result['drafts']);return result
    @app.post('/api/review/{id}')
    def review(id:str,body:dict=Body(...)):
        store.review(id,body.get('action'),body.get('question'));return {'ok':True}
    @app.get('/api/export')
    def export():
        return JSONResponse(store.export_bundle(),headers={'Content-Disposition':'attachment; filename=vla-practice-backup.json'})
    @app.post('/api/restore')
    async def restore(file:UploadFile=File(...)):
        data,suffix,name=await upload(file)
        try:bundle=json.loads(data)
        except (ValueError,UnicodeError) as e:raise ValueError('无效的 JSON 备份') from e
        store.restore(bundle);return {'ok':True}
    @app.post('/api/progress/clear')
    def clear(body:dict=Body(...)):
        if body.get('confirm') is not True:raise ValueError('需要确认清除')
        store.clear_progress();return {'ok':True}
    app.mount('/assets',StaticFiles(directory=assets),name='assets')
    out=ROOT/'web/out'
    @app.get('/{path:path}')
    def frontend(path:str):
        if path.startswith('api/'):raise HTTPException(404)
        target=(out/path).resolve()
        if not target.is_relative_to(out.resolve()):raise HTTPException(404)
        if target.is_file():return FileResponse(target)
        if (out/'index.html').exists():return FileResponse(out/'index.html')
        return JSONResponse({'detail':'前端尚未构建，请在 web 运行 npm run build'},503)
    return app

app=create_app()
