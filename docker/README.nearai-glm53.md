# GLM-5.3-Flash upstream integration candidate

This branch starts from official SGLang main `320bdd1ee2d6aface704c53d4674fd91a671fa74` and the official GLM-5.3-Flash support branch `4f5b9a380ce81b1cdae916360b4022ab7742a33f` (upstream PR #36507). It includes the merged disconnect/deferred-abort fix #35255. No Phala scheduler patches are included.

The image uses a pinned official CUDA 13 runtime and rebuilds SGLang and all native extensions from this checkout. The base dependency versions are Torch 2.13.0+cu130, Transformers 5.12.1, FlashInfer 0.6.18 and TileLang 0.1.12.

Build from a clean checkout:

```bash
docker build --build-arg SGLANG_REVISION="$(git rev-parse HEAD)" \
  -f docker/Dockerfile.nearai-glm53 -t sglang-glm53-upstream:local .
```

This is a qualification candidate, not a production release. Validation must include strict idle pool accounting, long chunked-prefill cancellation, multimodal cancellation, and text/vision/tool/SSE semantics with the actual serving settings. Upstream PR #34153 is cherry-picked: two final-prefill abort regression tests failed without it and passed with it. The integration also defers two context-parallel imports to avoid import cycles and preserves the OpenAI thinking_token_budget adapter (8192 default only for active `glm5_next`/`glm5_next_text` models, explicit null opt-out, native custom_params.thinking_budget takes precedence).

Review corrections clear freed Mamba tracking buffers on `ReqKvInfo`, use that owner for DSA tail transfer rows, include draft DSA index/tail buffers with hybrid targets, and dispatch encoder videos individually by representation. CPU regression checks cover allocation ownership, request conversion, encoder preprocessing, and transfer payload registration; these do not qualify distributed encoder or PD transport end to end.
