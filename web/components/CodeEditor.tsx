'use client';
import Editor, {loader} from '@monaco-editor/react';
loader.config({paths:{vs:'/monaco/vs'}});
export default function CodeEditor({value,onChange}:{value:string;onChange:(s:string)=>void}) {return <Editor height="480px" language="python" theme="vs-light" value={value} onChange={v=>onChange(v??'')} loading={<div className="empty">正在加载本地代码编辑器…</div>} options={{fontSize:14,minimap:{enabled:false},scrollBeyondLastLine:false,automaticLayout:true,tabSize:4,padding:{top:18},wordWrap:'on'}}/>}
