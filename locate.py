#!/usr/bin/env python3
"""Locate a target on the macOS desktop by recursive quadrant narrowing with a
Cloudflare Clef decision model, ending with a click-precision coordinate.

Each level: capture crops of the four quadrants of the current region, ask Clef
"which quadrant holds the target?" (choice) and "is the target visible?" (noul),
then narrow. When the region is small enough, ask "did we locate the item?".
"""

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

QUADS = ["top_left", "top_right", "bottom_left", "bottom_right"]
QUAD_LABELS = {
    "top_left": "top-left quadrant",
    "top_right": "top-right quadrant",
    "bottom_left": "bottom-left quadrant",
    "bottom_right": "bottom-right quadrant",
}
API_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def display_info():
    """Return (logical_width, logical_height, scale) for the main display."""
    try:
        data = json.loads(run(["system_profiler", "SPDisplaysDataType", "-json"]))
        for gpu in data["SPDisplaysDataType"]:
            for disp in gpu.get("spdisplays_ndrvs", []):
                if disp.get("spdisplays_main") != "spdisplays_yes":
                    continue
                logical = [int(v) for v in re.findall(r"\d+", disp["spdisplays_resolution"])][:2]
                pixels = [int(v) for v in re.findall(r"\d+", disp["_spdisplays_pixels"])][:2]
                if len(logical) == 2 and logical[0] > 0:
                    scale = max(1, round(pixels[0] / logical[0])) if len(pixels) == 2 else 1
                    return logical[0], logical[1], scale
    except Exception:
        pass
    return None, None, 1


def image_size(path):
    out = run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)])
    dims = {}
    for line in out.splitlines():
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].isdigit():
            dims[parts[0]] = int(parts[1])
    return dims["pixelWidth"], dims["pixelHeight"]


def capture_screen(path):
    subprocess.run(["screencapture", "-x", str(path)], check=True)


def crop(src, dst, x, y, w, h):
    run(["sips", "--cropOffset", str(y), str(x), "-c", str(h), str(w), str(src), "--out", str(dst)])


def encode_jpeg(src, dst, max_px, quality):
    run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(quality),
         "-Z", str(max_px), str(src), "--out", str(dst)])


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def payload_images(paths, max_px, quality):
    """Encode images for the Clef API; degrade quality until size limits hold."""
    images = []
    for i, src in enumerate(paths):
        dst = Path(src).with_suffix(".jpg")
        q = quality
        while True:
            encode_jpeg(src, dst, max_px, q)
            if dst.stat().st_size <= 4 * 1024 * 1024 or q <= 20:
                break
            q = max(20, q - 20)
        images.append({"content_type": "image/jpeg", "base64": b64(dst)})
    return images, [Path(p).with_suffix(".jpg") for p in paths]


def quadrant_crops(src, region, workdir, level, overlap=0.25):
    x, y, w, h = region
    half_w, half_h = w // 2, h // 2
    ox, oy = int(half_w * overlap), int(half_h * overlap)
    boxes = {
        "top_left": (x, y, min(w, half_w + ox), min(h, half_h + oy)),
        "top_right": (x + half_w - ox, y, min(w - half_w + ox, w), min(h, half_h + oy)),
        "bottom_left": (x, y + half_h - oy, min(w, half_w + ox), min(h - half_h + oy, h)),
        "bottom_right": (x + half_w - ox, y + half_h - oy,
                         min(w - half_w + ox, w), min(h - half_h + oy, h)),
    }
    paths = []
    for quad, (qx, qy, qw, qh) in boxes.items():
        dst = workdir / f"L{level}_{quad}.png"
        crop(src, dst, qx, qy, qw, qh)
        paths.append(dst)
    return paths, boxes


def ask(url, token, model, state, questions, images, timeout=600):
    body = json.dumps({
        "model": model,
        "state": state,
        "questions": questions,
        "images": images,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers)
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.load(resp)
    elapsed_ms = int((time.time() - start) * 1000)
    result = raw.get("result", raw) if isinstance(raw, dict) else raw
    if raw.get("success") is False:
        raise RuntimeError(f"API error: {json.dumps(raw)[:500]}")
    return result.get("answers", {}), result.get("usage", {}), elapsed_ms


def ask_local(state, questions, images, base_url, model, api_key):
    """OpenAI-compatible backend (vLLM/sglang/llama.cpp/Ollama). One chat call per
    question; probabilities recovered from first-token logprobs, text-parsed as fallback."""
    url = base_url.rstrip("/") + "/chat/completions"
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img["base64"]}}
               for img in images]
    answers = {}
    usage_in = usage_out = 0
    start = time.time()

    def chat(prompt, top):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}],
            "temperature": 0,
            "max_tokens": 8,
            "logprobs": True,
            "top_logprobs": top,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        if "error" in data:
            raise RuntimeError(str(data["error"])[:300])
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return choice, usage.get("prompt_tokens", -1), usage.get("completion_tokens", -1)

    for qid, q in questions.items():
        prompt = f"{state}\n\n{q['instructions']}"
        if q["type"] == "choice":
            keys = list(q["criteria"].keys())
            letters = "ABCDEFGH"[: len(keys)]
            prompt += ("\nAnswer with a single letter:\n"
                       + "\n".join(f"{l} = {k}" for l, k in zip(letters, keys)))
            choice_obj, in_t, out_t = chat(prompt, 20)
            usage_in += in_t
            usage_out += out_t
            tops = (choice_obj.get("logprobs") or {}).get("content") or []
            found = {}
            if tops:
                for entry in tops[0].get("top_logprobs", []):
                    tok = entry["token"].strip().upper()
                    if tok in letters and tok not in found:
                        found[tok] = entry["logprob"]
            probs = None
            if found:
                exps = {l: math.exp(v) for l, v in found.items()}
                total = sum(exps.values())
                probs = {l: exps[l] / total for l in letters}
            else:
                text = (choice_obj.get("message") or {}).get("content", "")
                m = re.search(rf"\b([{letters}])\b", text.upper())
                if m:
                    probs = {l: (1.0 if l == m.group(1) else 0.0) for l in letters}
            if probs:
                best = max(probs, key=probs.get)
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[letters.index(best)],
                    "probabilities": {keys[letters.index(l)]: p for l, p in probs.items()},
                    "confidence": probs[best],
                }
            else:
                answers[qid] = {"type": "choice", "choice": keys[0], "probabilities": {}, "confidence": 0.0}
        else:
            prompt += "\nAnswer with a single word: yes or no."
            choice_obj, in_t, out_t = chat(prompt, 20)
            usage_in += in_t
            usage_out += out_t
            tops = (choice_obj.get("logprobs") or {}).get("content") or []
            lp_yes = lp_no = None
            if tops:
                for entry in tops[0].get("top_logprobs", []):
                    tok = entry["token"].strip().lower()
                    if tok == "yes" and lp_yes is None:
                        lp_yes = entry["logprob"]
                    elif tok == "no" and lp_no is None:
                        lp_no = entry["logprob"]
            if lp_yes is not None and lp_no is not None:
                p_yes = math.exp(lp_yes) / (math.exp(lp_yes) + math.exp(lp_no))
            elif lp_yes is not None:
                p_yes = 1 / (1 + math.exp(-lp_yes))
            elif lp_no is not None:
                p_yes = 1 - 1 / (1 + math.exp(-lp_no))
            else:
                text = ((choice_obj.get("message") or {}).get("content") or "").strip().lower()
                p_yes = 1.0 if text.startswith("yes") else 0.0 if text.startswith("no") else 0.5
            answers[qid] = {"type": "noul", "noul": p_yes}

    return answers, {"input_tokens": usage_in, "output_tokens": usage_out}, int((time.time() - start) * 1000)


def mock_ask(state, questions, images):
    """Deterministic stand-in for the API used with --dry-run (clockwise drift)."""
    cycle = ["top_right", "bottom_right", "bottom_left", "top_left"]
    answers = {}
    n = mock_ask.calls
    mock_ask.calls += 1
    for qid, q in questions.items():
        if q["type"] == "choice":
            winner = cycle[n % 4]
            answers[qid] = {
                "type": "choice", "choice": winner,
                "probabilities": {k: 0.25 for k in QUADS},
                "confidence": 0.25,
            }
            answers[qid]["probabilities"][winner] = 0.4
        elif q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.75 if n >= 2 else 0.9}
    return answers, {"input_tokens": -1, "output_tokens": -1}, 0


mock_ask.calls = 0


def bar(p, width=24):
    filled = int(round(p * width))
    return "#" * filled + "-" * (width - filled)


def click_at(x, y):
    if subprocess.run(["which", "cliclick"], capture_output=True).returncode == 0:
        subprocess.run(["cliclick", f"c:{x},{y}"], check=True)
        return "cliclick"
    script = (
        'ObjC.import("CoreGraphics");'
        "var p = {x: %d, y: %d};"
        "var move = $.CGEventCreateMouseEvent($(), 5, p, 0);"
        "$.CGEventPost(0, move);"
        "var down = $.CGEventCreateMouseEvent($(), 1, p, 0);"
        "$.CGEventPost(0, down);"
        "var up = $.CGEventCreateMouseEvent($(), 2, p, 0);"
        "$.CGEventPost(0, up);" % (x, y)
    )
    subprocess.run(["osascript", "-l", "JavaScript", "-e", script], check=True)
    return "osascript"


def main():
    ap = argparse.ArgumentParser(description="Find a target on screen via recursive quadrant narrowing with Clef.")
    ap.add_argument("target", help="What to look for, e.g. 'the Safari reload button'")
    ap.add_argument("--model", default="clef-flash", choices=["clef-flash", "clef"])
    ap.add_argument("--base-url", help="Self-hosted OpenAI-compatible endpoint (e.g. http://host:8000/v1); skips Workers AI")
    ap.add_argument("--local-model", default="clef-flash", help="Model name to send to the local server")
    ap.add_argument("--api-key", default="EMPTY", help="API key for the local server")
    ap.add_argument("--url", help="SystemOne/Jev endpoint that accepts the native schema API "
                                  "(e.g. http://127.0.0.1:8790/v1/systemone from serve_local.py)")
    ap.add_argument("--depth", type=int, default=8, help="Max narrowing levels")
    ap.add_argument("--stop-px", type=int, default=48, help="Stop when region is at most this many logical px wide")
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="Bail if the location choice confidence drops below this")
    ap.add_argument("--img-px", type=int, default=768, help="Max dimension of uploaded images")
    ap.add_argument("--mode", default="full", choices=["full", "crops"],
                    help="full: one image of the current view per level; crops: four quadrant images")
    ap.add_argument("--image", help="Use an existing image instead of capturing the screen")
    ap.add_argument("--save", help="Directory to save crops and trace")
    ap.add_argument("--click", action="store_true", help="Send a mouse click at the final coordinates")
    ap.add_argument("--dry-run", action="store_true", help="Skip the API; use a mock decider to exercise the loop")
    args = ap.parse_args()

    workdir = Path(args.save) if args.save else Path("/var/folders/xj/3zs2dr3975712tszltvmbs1m0000gp/T/opencode/clef-desktop")
    workdir.mkdir(parents=True, exist_ok=True)

    logical_w, logical_h, scale = display_info()
    if args.image:
        full = Path(args.image).resolve()
    else:
        full = workdir / "screen.png"
        print(f"Capturing screen...")
        capture_screen(full)
    px_w, px_h = image_size(full)
    if args.image:
        scale = 1
        logical_w, logical_h = px_w, px_h
    elif logical_w is None:
        scale = max(1, round(px_w / 1800))
        logical_w, logical_h = px_w // scale, px_h // scale
    print(f"Target: {args.target!r}")
    print(f"Capture: {full.name} {px_w}x{px_h}px | display {logical_w}x{logical_h}pt | scale {scale}x")
    local = args.base_url is not None
    systemone_url = args.url
    if systemone_url:
        backend = f"systemone {systemone_url}"
    elif local:
        backend = "local " + args.local_model
    else:
        backend = args.model
    print(f"Model: {'MOCK' if args.dry_run else backend}\n")

    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_AUTH_TOKEN", "")
    url = systemone_url or API_URL.format(account=account, model=args.model)
    if not args.dry_run and not local and not systemone_url and (not account or not token):
        sys.exit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (or CLOUDFLARE_AUTH_TOKEN), "
                 "or pass --url / --base-url to use a self-hosted server.")

    def asker(state, questions, images):
        if args.dry_run:
            return mock_ask(state, questions, images)
        if local:
            return ask_local(state, questions, images, args.base_url, args.local_model, args.api_key)
        return ask(url, token if not systemone_url else "", args.model, state, questions, images)

    region = (0, 0, px_w, px_h)
    trace = []
    final = None

    for level in range(1, args.depth + 1):
        x, y, w, h = region
        half_w, half_h = w // 2, h // 2
        boxes = {
            "top_left": (x, y, half_w, half_h),
            "top_right": (x + half_w, y, w - half_w, half_h),
            "bottom_left": (x, y + half_h, half_w, h - half_h),
            "bottom_right": (x + half_w, y + half_h, w - half_w, h - half_h),
        }
        region_logical = (region[2] / scale, region[3] / scale)
        if args.mode == "full":
            full_crop = workdir / f"L{level}_region.png"
            crop(full, full_crop, x, y, w, h)
            upload_px = args.img_px
            if max(region_logical) > 1200:
                upload_px = max(args.img_px, 1536)
            images, jpgs = payload_images([full_crop], upload_px, 80)
            state = (
                f"You are helping a desktop automation agent locate: {args.target}.\n"
                f"The image is the current view, a {region_logical[0]:.0f}x{region_logical[1]:.0f}pt "
                "zoomed region of the screen."
            )
        else:
            crops, boxes = quadrant_crops(full, region, workdir, level)
            images, jpgs = payload_images(crops, args.img_px, 80)
            state = (
                f"You are helping a desktop automation agent locate: {args.target}.\n"
                f"The four images are the four quadrants of the current view, which is a "
                f"{region_logical[0]:.0f}x{region_logical[1]:.0f}pt zoomed region of the screen. "
                "Image 1 is the top-left quadrant, image 2 the top-right, image 3 the bottom-left, "
                "image 4 the bottom-right. Adjacent quadrants overlap slightly at their edges."
            )
        questions = {
            "where": {
                "type": "choice",
                "instructions": "Which quadrant contains the target?",
                "criteria": {
                    "top_left": "Upper-left quarter of the view",
                    "top_right": "Upper-right quarter of the view",
                    "bottom_left": "Lower-left quarter of the view",
                    "bottom_right": "Lower-right quarter of the view",
                },
            },
        }
        answers, usage, ms = asker(state, questions, images)
        where = answers.get("where", {})
        choice = where.get("choice", "top_left")
        probs = where.get("probabilities", {})
        confidence = probs.get(choice, 0.0)
        line = f"L{level} region={region_logical[0]:.0f}x{region_logical[1]:.0f}pt {ms:>5}ms conf={confidence:.2f} -> {choice}"
        for q in QUADS:
            line += f"\n    {q:<13} {bar(probs.get(q, 0))} {probs.get(q, 0):.2f}"
        print(line)

        qx, qy, qw, qh = boxes[choice]
        trace.append({"level": level, "region_px": list(region), "choice": choice,
                      "probabilities": probs, "confidence": confidence, "ms": ms, "usage": usage})
        region = (qx, qy, qw, qh)

        if confidence < args.min_conf:
            print(f"\nLocation confidence {confidence:.2f} below {args.min_conf}; stopping before losing the target.")
            break
        if region[2] / scale <= args.stop_px or region[3] / scale <= args.stop_px:
            break

    final_x, final_y, final_w, final_h = region
    view_w = max(3 * final_w, min(256, px_w))
    view_h = max(3 * final_h, min(256, px_h))
    view_x = max(0, min(px_w - view_w, final_x + final_w // 2 - view_w // 2))
    view_y = max(0, min(px_h - view_h, final_y + final_h // 2 - view_h // 2))
    final_crop = workdir / "final.png"
    crop(full, final_crop, view_x, view_y, view_w, view_h)
    final_jpg = Path(str(final_crop)).with_suffix(".jpg")
    encode_jpeg(final_crop, final_jpg, args.img_px, 90)

    state = (
        f"You are a desktop automation agent verifying a target was found: {args.target}.\n"
        "The single image shows the region the search converged on, plus surrounding context."
    )
    questions = {
        "located_quad": {
            "type": "choice",
            "instructions": "Which quadrant contains the target?",
            "criteria": {
                "top_left": "Upper-left quarter of the view",
                "top_right": "Upper-right quarter of the view",
                "bottom_left": "Lower-left quarter of the view",
                "bottom_right": "Lower-right quarter of the view",
            },
        },
        "located_id": {
            "type": "choice",
            "instructions": "What is in the middle of this image?",
            "criteria": {
                "target": f"The {args.target}",
                "other": "A different UI element or content",
                "nothing": "No distinct UI element there",
            },
        },
    }
    if args.dry_run:
        answers, usage, ms = mock_ask(state, questions, [])
    else:
        images, _ = payload_images([final_jpg], args.img_px, 90)
        answers, usage, ms = asker(state, questions, images)
    quad = answers.get("located_quad", {})
    quad_probs = quad.get("probabilities", {})
    quad_prob = max(quad_probs.values()) if quad_probs else 0.0
    identity = answers.get("located_id", {})
    id_probs = identity.get("probabilities", {})
    id_prob = id_probs.get("target", 0.0)
    located = max(quad_prob, id_prob)
    print(f"\nFINAL  {ms:>5}ms did_we_locate={located:.2f} "
          f"(quadrant {quad.get('choice')} {quad_prob:.2f} | identity {identity.get('choice')} {id_prob:.2f})")
    for q in QUADS:
        print(f"    {q:<13} {bar(quad_probs.get(q, 0))} {quad_probs.get(q, 0):.2f}")
    trace.append({"level": "final", "region_px": list(region), "located": located,
                  "quadrant_prob": quad_prob, "identity_prob": id_prob, "ms": ms, "usage": usage})

    click_x = (final_x + final_w / 2) / scale
    click_y = (final_y + final_h / 2) / scale
    summary = {
        "target": args.target,
        "model": args.local_model if local else args.model,
        "located_probability": located,
        "click_point": [round(click_x), round(click_y)],
        "region_logical": [round(final_x / scale), round(final_y / scale),
                           round(final_w / scale), round(final_h / scale)],
        "levels": len(trace) - 1,
        "total_ms": sum(t["ms"] for t in trace),
    }
    if args.save:
        (workdir / "trace.json").write_text(json.dumps(trace, indent=2))
    print("\n" + json.dumps(summary, indent=2))

    if args.click and located >= args.min_conf:
        method = click_at(round(click_x), round(click_y))
        print(f"Clicked ({round(click_x)},{round(click_y)}) via {method}")
    elif args.click:
        print("Confidence below threshold; no click sent.")


if __name__ == "__main__":
    main()
