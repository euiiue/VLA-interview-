import type { Metadata } from 'next';
import './globals.css';
import 'katex/dist/katex.min.css';
export const metadata:Metadata={title:'VLA 面试练习室',description:'从理解到实现，循序训练视觉语言动作模型与机器人学习。'};
export default function RootLayout({children}:{children:React.ReactNode}) {return <html lang="zh-CN"><body>{children}</body></html>}
