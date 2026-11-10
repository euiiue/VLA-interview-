import { errorMessage } from './practice';
export async function api<T>(path:string, body?:unknown, method='POST'):Promise<T> {
 const response = await fetch('/api'+path, body === undefined ? {cache:'no-store'} : {
  method, headers:{'X-VLA-Request':'1', ...(body instanceof FormData ? {} : {'Content-Type':'application/json'})},
  body:body instanceof FormData ? body : JSON.stringify(body)
 });
 const json = await response.json().catch(()=>({detail:`服务响应异常（${response.status}）`}));
 if(!response.ok) throw new Error(errorMessage(json));
 return json as T;
}
