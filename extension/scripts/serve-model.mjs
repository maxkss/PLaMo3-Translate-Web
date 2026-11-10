// Serves the converted model directory with CORS (+ CORP for cross-origin-isolated extension pages).
// Usage: npm run serve-model [-- <dir>] [--port 8787]
import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { join, normalize, resolve } from "node:path";

const args = process.argv.slice(2);
const portIdx = args.indexOf("--port");
const port = portIdx >= 0 ? Number(args.splice(portIdx, 2)[1]) : 8787;
const root = resolve(args[0] ?? new URL("../../models/onnx/plamo-3-translate-2b", import.meta.url).pathname);

createServer((req, res) => {
  const headers = {
    "Access-Control-Allow-Origin": "*",
    "Cross-Origin-Resource-Policy": "cross-origin",
    "Cache-Control": "no-store",
  };
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^(\.\.[/\\])+/, "");
  const file = join(root, path);
  let st;
  try {
    st = statSync(file);
  } catch {}
  if (!file.startsWith(root) || !st?.isFile()) {
    res.writeHead(404, headers).end("not found");
    return;
  }
  res.writeHead(200, { ...headers, "Content-Type": "application/octet-stream", "Content-Length": st.size });
  if (req.method === "HEAD") return res.end();
  createReadStream(file).pipe(res);
  console.log(req.method, path, st.size);
}).listen(port, () => console.log(`serving ${root} at http://localhost:${port}/`));
