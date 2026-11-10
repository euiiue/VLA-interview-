export type Option = {id:string;text:string};
export type Question = {id:string;source_id:string;chapter:string;type:'single'|'multiple'|'cloze';stem:string;options:Option[];version:number;reviewed:boolean;[key:string]:unknown};
export type Attempt = {id:string;question_id:string;version:number;answer:string[];correct:boolean;created_at:string};
export type Source = {id:string;number:number;title:string;chapter:string;chapter_title:string;answer:string;images:{src:string;caption?:string}[];citations:unknown[]};
export type Exercise = {id:string;source_id:string;number:number;title:string;chapter:string;difficulty:string;description:string;starter:string;hints:string[];test_names:string[]};
export type Environment = {docker_available:boolean;image_available:boolean;code_ready:boolean;image:string;python:string;ai_configured:boolean;ai_provider:string;ai_model:string;message:string};
export type Bootstrap = {sources:Source[];questions:Question[];exercises:Exercise[];stats:{total:number;attempted:number;correct:number;wrong:number;accuracy:number;by_chapter:{chapter:string;total:number;attempted:number;correct:number}[];code_runs:number|CodeRun[]};environment:Environment;favorites:string[];attempts:Attempt[];code_drafts:Record<string,string>;review_drafts:Question[]};
export type Result = {correct:boolean;expected:(string|string[])[];explanation:string;attempt:Attempt};
export type RunResult = {status:'passed'|'failed'|'timeout'|'unavailable'|'error';cases:{name:string;passed:boolean;error?:string}[];stdout:string;elapsed_ms:number;message?:string};
export type Preview = {token:string;filename:string;duplicate:boolean;sources:Source[];questions:Question[];warnings:string[]};

export type CodeRun = RunResult & {id:string;exercise_id:string;created_at:string};
