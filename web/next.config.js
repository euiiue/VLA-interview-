/** @type {import('next').NextConfig} */
const nextConfig = {
 output: 'export', trailingSlash: true, reactStrictMode: true,
 images: { unoptimized: true }, experimental: { cpus: 2 },
 ...(process.env.NODE_ENV === 'development' ? { async rewrites() {
 return [{source:'/api/:path*',destination:`${process.env.VLA_API_BASE || 'http://127.0.0.1:8765'}/api/:path*`},{source:'/assets/:path*',destination:`${process.env.VLA_API_BASE || 'http://127.0.0.1:8765'}/assets/:path*`}];
 }} : {})
};
module.exports = nextConfig;
