from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "app" / "web"
WEB_DIST = WEB_ROOT / "dist" / "index.html"


def _frontend_needs_build() -> bool:
    if not WEB_DIST.exists():
        return True
    inputs = [WEB_ROOT / "package.json", WEB_ROOT / "package-lock.json", WEB_ROOT / "vite.config.js"]
    inputs.extend((WEB_ROOT / "src").rglob("*"))
    newest_input = max(path.stat().st_mtime for path in inputs if path.is_file())
    return newest_input > WEB_DIST.stat().st_mtime


def _build_frontend() -> None:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise RuntimeError("Node.js 20.19 이상이 필요합니다. https://nodejs.org 에서 설치해 주세요.")
    if not (WEB_ROOT / "node_modules").exists():
        print("[SignAid] 웹 의존성을 설치합니다…", flush=True)
        subprocess.run([npm, "ci"], cwd=WEB_ROOT, check=True)
    print("[SignAid] 웹 화면을 빌드합니다…", flush=True)
    subprocess.run([npm, "run", "build"], cwd=WEB_ROOT, check=True)


def main() -> None:
    cli = argparse.ArgumentParser(description="SignAid 웹 앱과 API를 한 번에 실행합니다.")
    cli.add_argument("--host", default="127.0.0.1", help="바인딩 주소 (기본값: 127.0.0.1)")
    cli.add_argument("--port", type=int, default=8000, help="포트 (기본값: 8000)")
    cli.add_argument("--no-build", action="store_true", help="웹 빌드 확인을 건너뜁니다.")
    cli.add_argument("--no-open", action="store_true", help="브라우저를 자동으로 열지 않습니다.")
    cli.add_argument("--reload", action="store_true", help="개발용 자동 재시작을 사용합니다.")
    args = cli.parse_args()

    if not 1 <= args.port <= 65535:
        cli.error("port must be between 1 and 65535")

    try:
        if not args.no_build and _frontend_needs_build():
            _build_frontend()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[SignAid] 시작 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not WEB_DIST.exists():
        print("[SignAid] 웹 빌드가 없습니다. app/web에서 npm run build를 먼저 실행해 주세요.", file=sys.stderr)
        raise SystemExit(1)

    import uvicorn

    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{args.port}"
    if not args.no_open and not args.reload:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"[SignAid] {url} 에서 실행합니다. 종료: Ctrl+C", flush=True)
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
