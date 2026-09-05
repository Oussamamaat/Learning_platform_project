IBLOG Tutor — RTX 5090 benchmark  (2026-09-04T08:50:04Z)
base_url=http://localhost:8000
## GPU / CUDA
── nvidia-smi ──
  name: Quadro RTX 8000
  driver_version: 580.159.04
  memory_total_mib: 49152
  memory_free_mib: 29756
  memory_used_mib: 18646

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
{"nvidia_smi": {"name": "Quadro RTX 8000", "driver_version": "580.159.04", "memory_total_mib": 49152, "memory_free_mib": 29756, "memory_used_mib": 18646}, "torch": {"version": "2.11.0+cu128", "cuda_available": true, "cuda_build": "12.8", "device_name": "Quadro RTX 8000", "compute_capability": "7.5", "total_vram_gib": 47.3, "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]}, "live_op": "ok"}
## Service health
{"status":"ok","version":"0.1.0"}
```
NAME                     ID              SIZE      MODIFIED          
iblog-tutor-fr:latest    e34aa445f2db    5.8 GB    About an hour ago    
IBLOG_TUTOR:latest       a147f1f44493    5.8 GB    About an hour ago    
```
## LLM latency / throughput
── chat turns (production path: retrieval + grounding + LLM) ──
  [darija] turn 1:  2.57s  (253 chars)
  [darija] turn 2:  2.64s  (239 chars)
  [darija] turn 3:  2.69s  (251 chars)
  [french] turn 1:  2.00s  (228 chars)
  [french] turn 2:  2.32s  (282 chars)
  [french] turn 3:  3.06s  (405 chars)
  darija mean: 2.63s
  french mean: 2.46s

── language-switch cost (alternating fr ↔ darija) ──
  switch turn 1 (ar-MA):  3.29s
  switch turn 2 (fr):  3.29s
  switch turn 3 (ar-MA):  2.67s
  switch turn 4 (fr):  3.45s
  switch turn 5 (ar-MA):  3.01s
  switch turn 6 (fr):  2.48s

── raw Ollama generation (IBLOG_TUTOR:latest) ──
  TTFT: 9.21s   tokens: 86   8.1 tok/s

Wrote benchmark_llm.json
## Voice pipeline
SKIPPED (set VOICE_WAV=... and STT_ENGINE/TTS_ENGINE off 'none')
## OCR
SKIPPED (set OCR_IMAGE=/path/to/page.png)
## Peak VRAM during run
peak GPU memory used: 18894 MiB
