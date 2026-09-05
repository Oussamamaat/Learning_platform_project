IBLOG Tutor — RTX 5090 benchmark  (2026-09-04T10:36:25Z)
base_url=http://localhost:8000

## GPU / CUDA
── nvidia-smi ──
  name: Quadro RTX 8000
  driver_version: 580.159.04
  memory_total_mib: 49152
  memory_free_mib: 30098
  memory_used_mib: 18304

── torch.cuda ──
  torch 2.11.0+cu128 (built for CUDA 12.8)
  cuda_available: True
  device: Quadro RTX 8000
  compute capability: 7.5  (RTX 5090 = 12.0)
  total VRAM: 47.3 GiB
  torch arch_list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']

── live CUDA op (matmul on device) ──
  OK — matmul ran on the GPU

RESULT: GPU usable ✓
{"nvidia_smi": {"name": "Quadro RTX 8000", "driver_version": "580.159.04", "memory_total_mib": 49152, "memory_free_mib": 30098, "memory_used_mib": 18304}, "torch": {"version": "2.11.0+cu128", "cuda_available": true, "cuda_build": "12.8", "device_name": "Quadro RTX 8000", "compute_capability": "7.5", "total_vram_gib": 47.3, "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]}, "live_op": "ok"}

## Service health
{"status":"ok","version":"0.1.0"}

## LLM latency / throughput
── chat turns (production path: retrieval + grounding + LLM) ──
  [darija] turn 1:  2.95s  (270 chars)
  [darija] turn 2:  3.07s  (330 chars)
  [darija] turn 3:  3.28s  (333 chars)
  [french] turn 1:  3.38s  (484 chars)
  [french] turn 2:  2.32s  (281 chars)
  [french] turn 3:  2.47s  (296 chars)
  darija mean: 3.10s
  french mean: 2.72s

── language-switch cost (alternating fr ↔ darija) ──
  switch turn 1 (ar-MA):  2.91s
  switch turn 2 (fr):  3.38s
  switch turn 3 (ar-MA):  3.42s
  switch turn 4 (fr):  2.72s
  switch turn 5 (ar-MA):  2.13s
  switch turn 6 (fr):  2.94s

── raw Ollama generation (IBLOG_TUTOR:latest) ──
  TTFT: 8.68s   tokens: 86   8.5 tok/s

Wrote benchmark_llm.json

## Voice pipeline
connecting: ws://localhost:8000/api/v1/voice/session
  transcript: 'سوف نقوم بإعطاء الشخصية التي يجب أن نلتقي بها'
Wrote benchmark_voice.json

## OCR
SKIPPED (set OCR_IMAGE=/path/to/page.png)

## Peak VRAM during run
peak GPU memory used: 21366 MiB

Report written to benchmark_report.md (+ benchmark_*.json).
