"""AquariusCodaExp 训练监控前端（stdlib only，无第三方依赖）。

用法：python dashboard.py [--port 8791]
- /         → static/dashboard.html
- /metrics  → runs/*/metrics.jsonl 全量解析（step / eval / save 三类）
- /log      → train.log 末尾 60 行
"""
import argparse
import glob
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
RUN = None


_CACHE = {"sig": None, "offsets": {}, "steps": {}, "evals": {}, "saves": {}, "samples": {}, "run_dir": None, "json": None}
MAX_STEPS_POINTS = 1500  # 响应中 step 行的降采样上限，防止长跑后轮询载荷膨胀


def _parse_line(obj, steps, evals, saves, samples):
    t = obj.get("type")
    k = obj.get("step")
    if k is None:
        return
    if t == "step":
        steps[k] = obj
    elif t == "eval":
        evals[k] = obj
    elif t == "save":
        saves[k] = obj
    elif t == "sample":
        samples[k] = obj


def load_metrics(run=None):
    """合并 run 目录下全部 metrics*.jsonl（含归档 metrics_part1.jsonl），增量解析。

    - 断点续训后曲线仍从 step 0 开始：归档与新文件按 step 去重，新文件优先；
    - 增量解析：每文件只读新增字节（记录 offset），轮询成本 O(新增行数)；
    - step 行超过 1500 时按序降采样（始终保留最后一个点），前端用线性步数轴还原真实间距。
    返回缓存的 JSON 字符串（内容有变化时才重新生成）。
    """
    base_runs = os.path.join(BASE, "runs")
    if run:
        run_dir = os.path.join(base_runs, run)
    else:
        dirs = [d for d in glob.glob(os.path.join(base_runs, "*")) if os.path.isdir(d)]
        if not dirs:
            return '{"steps":[],"evals":[],"saves":[],"samples":[]}'
        run_dir = max(dirs, key=os.path.getmtime)
    paths = sorted(glob.glob(os.path.join(run_dir, "metrics*.jsonl")))
    sig = tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in paths)

    if _CACHE["sig"] != sig:
        for p in paths:
            off = _CACHE["offsets"].get(p, 0) if _CACHE.get("run_dir") == run_dir else 0
            size = os.path.getsize(p)
            if size < off:
                off = 0
            if size == off:
                continue
            with open(p, "r", encoding="utf-8") as f:
                f.seek(off)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _parse_line(json.loads(line), _CACHE["steps"], _CACHE["evals"], _CACHE["saves"], _CACHE["samples"])
                    except json.JSONDecodeError:
                        pass
                _CACHE["offsets"] = {**_CACHE["offsets"], p: f.tell()}
        _CACHE["sig"] = sig
        _CACHE["run_dir"] = run_dir
        _CACHE["json"] = None

    if _CACHE["json"] is None:
        step_list = [_CACHE["steps"][k] for k in sorted(_CACHE["steps"])]
        if len(step_list) > MAX_STEPS_POINTS:
            stride = -(-len(step_list) // MAX_STEPS_POINTS)
            step_list = step_list[::stride] + [step_list[-1]]
        data = {
            "steps": step_list,
            "evals": [_CACHE["evals"][k] for k in sorted(_CACHE["evals"])],
            "saves": [_CACHE["saves"][k] for k in sorted(_CACHE["saves"])],
            "samples": [_CACHE["samples"][k] for k in sorted(_CACHE["samples"])],
        }
        _CACHE["json"] = json.dumps(data, ensure_ascii=False)
    return _CACHE["json"]


def tail_log(n=60):
    out = []
    for path in sorted(glob.glob(os.path.join(BASE, "runs", "*", "*.log")), key=os.path.getmtime):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            out.extend(lines[-n:])
        except OSError:
            pass
    return "".join(out[-n:])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/metrics":
            self._send(200, load_metrics(RUN))
        elif path == "/log":
            self._send(200, tail_log(), "text/plain; charset=utf-8")
        elif path == "/sample":
            cands = sorted(
                glob.glob(os.path.join(BASE, "runs", "*", "samples", "latest.png")),
                key=os.path.getmtime,
            )
            if RUN:
                cands = [c for c in cands if os.path.basename(os.path.dirname(os.path.dirname(c))) == RUN]
            if not cands:
                self._send(404, "no sample yet", "text/plain; charset=utf-8")
                return
            with open(max(cands, key=os.path.getmtime), "rb") as f:
                self._send(200, f.read(), "image/png")
        elif path in ("/", "/index.html"):
            with open(os.path.join(BASE, "static", "dashboard.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, '{"error":"not found"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--run", default=None, help="runs/ 下的子目录名，默认取最近修改的")
    args = ap.parse_args()
    global RUN
    RUN = args.run
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://127.0.0.1:{args.port} (run={args.run or 'latest'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
