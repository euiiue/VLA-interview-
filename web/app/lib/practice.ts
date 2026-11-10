import type {Attempt,Option,Question} from './types';
export type Filters={query?:string;chapter?:string;type?:string;status?:string};
export function latestAttempts(attempts:Attempt[]):Map<string,Attempt> {
 const map=new Map<string,Attempt>();
 for(const a of attempts) {const key=`${a.question_id}:${a.version}`;if(!map.has(key)||a.created_at>=map.get(key)!.created_at)map.set(key,a)}
 return map;
}
export function filterQuestions(questions:Question[],filters:Filters,attempts:Attempt[],favorites:string[]):Question[] {
 const latest=latestAttempts(attempts);const query=(filters.query??'').toLocaleLowerCase().trim();
 return questions.filter(q=>{
  const last=latest.get(`${q.id}:${q.version}`);
  return q.reviewed && (!filters.chapter||q.chapter===filters.chapter)&&(!filters.type||q.type===filters.type)&&(!query||`${q.stem} ${q.source_id} ${q.id}`.toLocaleLowerCase().includes(query))&&(!filters.status||filters.status==='all'||(filters.status==='new'&&!last)||(filters.status==='wrong'&&last?.correct===false)||(filters.status==='mastered'&&last?.correct===true)||(filters.status==='favorites'&&favorites.includes(q.id)));
 });
}
export function shuffleOptions(options:Option[],random:()=>number=Math.random):Option[]{const copy=[...options];for(let i=copy.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[copy[i],copy[j]]=[copy[j],copy[i]]}return copy}
export function displayedAnswer(expected:(string|string[])[],options:Option[]):string {return expected.map(value=>{const id=String(value);const i=options.findIndex(option=>option.id===id);return i<0?id:String.fromCharCode(65+i)}).join('、')}
export function errorMessage(error:unknown):string {
 if(error instanceof Error)return error.message;
 if(typeof error==='string')return error;
 if(error&&typeof error==='object'&&'detail'in error){const detail=(error as {detail:unknown}).detail;if(typeof detail==='string')return detail;if(Array.isArray(detail))return detail.map(e=>`${e.loc?.join('.')??''} ${e.msg??JSON.stringify(e)}`).join('；')}
 return '操作失败，请检查本机服务后重试。';
}
