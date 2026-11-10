import { cp, mkdir } from 'node:fs/promises';
await mkdir('public/monaco', { recursive: true });
await cp('node_modules/monaco-editor/min/vs', 'public/monaco/vs', {recursive:true});
console.log('Monaco assets copied locally.');
