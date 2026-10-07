#!/usr/bin/env python3
"""Serve the local CLEF-Flash int8 ConvRot release over the Jev/SystemOne API on Apple Silicon.

Loads the packed mixed-int8 checkpoint with the release's own torch backend, bakes every
packed weight to bf16 (the release's own decode), moves to MPS, and exposes
POST /v1/systemone. No approximation: these are the shipped calibrated weights, decoded
by the official unpacker, computed in bf16 like the reference torch path.
"""

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from torch.nn.utils import parametrize


def bake_model(model, packed):
    """Materialize packed weights as plain bf16 parameters and free the packed buffers."""
    count = 0
    for name, module in list(model.named_modules()):
        leaf = name.rpartition(".")[2]
        parent_name = name.rpartition(".")[0]
        if isinstance(module, packed.PackedInputEmbedding):
            weight = module.packed()
            container = model.get_submodule(parent_name)
            embedding = torch.nn.Embedding(module.num_embeddings, module.embedding_dim,
                                           padding_idx=module.padding_idx, dtype=weight.dtype)
            with torch.no_grad():
                embedding.weight.copy_(weight)
            setattr(container, leaf, embedding)
            count += 1
        elif isinstance(module, packed.PackedOutputLookup):
            weight = module.packed()
            container = model.get_submodule(parent_name)
            linear = torch.nn.Linear(module.in_features, module.out_features,
                                     bias=False, dtype=weight.dtype)
            with torch.no_grad():
                linear.weight.copy_(weight)
            setattr(container, leaf, linear)
            count += 1
    for _, owner in list(model.named_modules()):
        parametrizations = getattr(owner, "parametrizations", None)
        if parametrizations is None:
            continue
        for leaf in list(parametrizations):
            parametrize.remove_parametrizations(owner, leaf, leave_parametrized=True)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="../cloudflare-clef-flash")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--skip-bake", action="store_true", help="Keep packed weights; decode per forward")
    ap.add_argument("--no-patches", action="store_true", help="Disable the performance patches")
    args = ap.parse_args()

    release = Path(args.release).resolve()
    sys.path.insert(0, str(release / "runtime"))
    sys.path.insert(0, str(release))

    import torch
    from torch.nn.utils import parametrize
    import clef_flash_packed as packed
    from joint_schema_model import systemone
    from PIL import Image

    import fast_patches
    if not args.no_patches:
        fast_patches.install()

    print("loading packed checkpoint (backend=torch, device=cpu)...", flush=True)
    start = time.time()
    model, processor = packed.load_checkpoint(release, device="cpu", backend="torch")
    print(f"checkpoint loaded in {time.time() - start:.1f}s", flush=True)

    if not args.skip_bake:
        start = time.time()
        count = bake_model(model, packed)
        print(f"baked {count} packed weights in {time.time() - start:.1f}s", flush=True)

    if not args.no_patches:
        fast_patches.patch_head_cpu_mask(model)
        print("performance patches active (metal row-substitution, sync removals)", flush=True)

    start = time.time()
    model.to(args.device)
    print(f"model on {args.device} in {time.time() - start:.1f}s", flush=True)

    lock = threading.Lock()

    def decode_images(raw):
        images = []
        for item in raw or []:
            if isinstance(item, str):
                header, _, data = item.partition(",")
                if not data:
                    data = header
                    header = ""
            else:
                data = item["base64"]
            images.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        return images

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *log_args):
            print(f"[http] {fmt % log_args}", flush=True)

        def respond(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ok", "device": args.device})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/systemone", "/systemone"):
                self.respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length))
            except Exception as error:
                self.respond(400, {"error": str(error)})
                return
            try:
                request = dict(request)
                request["images"] = decode_images(request.pop("images", None))
                with lock:
                    result = systemone(model, processor, request)
                self.respond(200, result)
            except Exception as error:
                self.respond(500, {"error": f"{type(error).__name__}: {error}"})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"systemone endpoint on http://{args.host}:{args.port}/v1/systemone", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
