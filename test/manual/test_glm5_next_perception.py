"""Pixel-dependent GLM vision gate against an already running server.

Example: python test/manual/test_glm5_next_perception.py --base-url http://localhost:30000
Use OPENAI_API_KEY for an authenticated endpoint. Fixtures are synthetic PNGs;
answers are never included in prompts. A failed case produces a nonzero exit.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
import zlib


def png_rows(size, row_fn):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\0" + row_fn(y) for y in range(size))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def fixtures(size=512):
    colors = {"red": (255, 0, 0), "green": (0, 160, 0), "blue": (0, 0, 255)}
    solid = {
        name: png_rows(size, lambda y, c=c: bytes(c) * size)
        for name, c in colors.items()
    }
    question = "What is the single dominant color of this image? One lowercase word."
    cases = [(f"solid-{name}", [data], question, name) for name, data in solid.items()]
    halves = png_rows(
        size,
        lambda y: (
            bytes(colors["red"]) * (size // 2) + bytes(colors["blue"]) * (size // 2)
        ),
    )
    cases.append(
        (
            "halves",
            [halves],
            "The image has two halves. What color is the LEFT half and what color is the RIGHT half? Answer as: left=<color>, right=<color>.",
            "halves",
        )
    )
    band = png_rows(
        size,
        lambda y: (
            bytes((255, 255, 255) if size // 3 < y < 2 * size // 3 else (0, 0, 0))
            * size
        ),
    )
    cases.append(
        (
            "band",
            [band],
            "Describe this image in one short sentence: what colors and what shape do you see?",
            "band",
        )
    )
    for count in (2, 4):
        cases.append(
            (
                f"count-{count}",
                [solid["red"], solid["blue"]] * (count // 2),
                "How many images are attached? Answer with a single digit.",
                str(count),
            )
        )
    return cases


def matches(output, expected):
    output = output.strip().lower().strip(" .\n")
    if expected == "halves":
        return (
            re.fullmatch(r"left\s*=\s*red\s*,\s*right\s*=\s*blue", output) is not None
        )
    if expected == "band":
        return (
            all(
                re.search(rf"\b{word}\b", output)
                for word in ("white", "black", "horizontal")
            )
            and re.search(r"\b(rectangle|band|stripe|bar)\b", output) is not None
            and "circle" not in output
        )
    return output == expected


def check(args, case):
    name, images, prompt, expected = case
    content = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode(),
                "detail": args.detail,
            },
        }
        for data in images
    )
    if args.image_first:
        content = content[1:] + content[:1]
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 1024,
        "thinking_token_budget": args.thinking_budget,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    result = {
        "case": name,
        "size": args.size,
        "image_sha256": [hashlib.sha256(data).hexdigest() for data in images],
        "ok": False,
    }
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            json.dumps(body).encode(),
            headers,
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.load(response)
            choice = payload["choices"][0]
            result.update(
                status=response.status,
                finish=choice["finish_reason"],
                prompt_tokens=payload.get("usage", {}).get("prompt_tokens"),
            )
            result["ok"] = (
                response.status == 200
                and choice["finish_reason"] == "stop"
                and matches(choice["message"].get("content") or "", expected)
            )
    except Exception as error:
        result["error"] = type(error).__name__
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    print(json.dumps(result), flush=True)
    return result["ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--model", default="z-ai/glm-5.3-flash")
    parser.add_argument("--size", type=int, choices=(256, 512, 1024, 2048), default=512)
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--thinking-budget", type=int, choices=(0, 512), default=0)
    parser.add_argument("--detail", choices=("auto", "high"), default="auto")
    parser.add_argument("--image-first", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda case: check(args, case), fixtures(args.size)))
    print(
        json.dumps(
            {"event": "finished", "passed": sum(results), "total": len(results)}
        ),
        flush=True,
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
