import { describe, expect, it } from "vitest";
import { displayedAnswer, latestAttempts, filterQuestions, shuffleOptions, errorMessage } from "./practice";
const questions = [
 {id:"a",source_id:"Q1",chapter:"1",type:"single",stem:"注意力",version:2,reviewed:true,options:[{id:"x",text:"X"},{id:"y",text:"Y"}]},
 {id:"b",source_id:"Q2",chapter:"2",type:"cloze",stem:"动作",version:1,reviewed:true,options:[]}
] as any;
describe("practice state", () => {
 it("does not count stale question versions as mastered", () => {
  const attempts = [{question_id:"a",version:1,correct:true,created_at:"2026-01-01"}] as any;
  expect(filterQuestions(questions, {status:"new"}, attempts, []).map(q=>q.id)).toEqual(["a","b"]);
 });
 it("resolves a wrong answer after the latest current-version answer is correct", () => {
  const attempts = [{question_id:"a",version:2,correct:false,created_at:"2026-01-01"},{question_id:"a",version:2,correct:true,created_at:"2026-01-02"}] as any;
  expect(filterQuestions(questions, {status:"wrong"}, attempts, [])).toEqual([]);
  expect(latestAttempts(attempts).get("a:2")?.correct).toBe(true);
 });
 it("combines chapter, query, type and favorites filters",()=> {
  expect(filterQuestions(questions,{chapter:"1",query:"Q1",type:"single",status:"favorites"},[],["a"]).map(q=>q.id)).toEqual(["a"]);
  expect(filterQuestions(questions,{query:"不存在"},[],[])).toEqual([]);
 });
 it("shuffles a copy with stable option ids",()=> {
  const result = shuffleOptions(questions[0].options,()=>0);
  expect(result.map(o=>o.id)).toEqual(["y","x"]);
  expect(questions[0].options.map((o:any)=>o.id)).toEqual(["x","y"]);
  expect(displayedAnswer(["x"], result)).toBe("B");
 });
 it("renders validation errors without crashing",()=> {
  expect(errorMessage({detail:[{loc:["body","answer"],msg:"字段必填"}]})).toContain("字段必填");
  expect(errorMessage({detail:"服务未就绪"})).toBe("服务未就绪");
 });
});
