"""
Wan2.2 renderer via the ComfyUI HTTP API.

Implements the 7-step TODO written out in the partner repo's
src/scene_planner/wan_client.py, whose generate_video() still raises
NotImplementedError. Lives here rather than there because that repo is the
partner's and is never edited from this one (settings.video_pipeline_dir);
their run_phase_render() already takes a `wan_generate_video=` injection
parameter, so this slots in as a caller-supplied renderer with no fork.

Drop-in signature-compatible with wan_client.generate_video and
wan_client.generate_video_mock, so scripts/video/worker.py chooses between
them purely on whether settings.comfyui_url is set.

Two generation branches, exactly as the partner's manual ComfyUI runs did:
  - no reference_image -> text->video (the first scene)
  - reference_image    -> image+text->video, seeded with the previous
    scene's last frame, which is what makes consecutive clips look like
    one continuous video rather than unrelated shots

Uses urllib from the stdlib, matching app/services/llm.py -- this repo
deliberately carries no HTTP client dependency.
"""
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

# A ComfyUI workflow exported through "Save (API Format)". Node ids are
# workflow-specific, so rather than hardcoding "node 6 is the prompt" we
# locate nodes by class_type and by which input we need to overwrite --
# that survives the partner re-exporting their workflow with a different
# node layout, which is the likeliest way this breaks.
_POSITIVE_PROMPT_CLASS = "CLIPTextEncode"
_LOAD_IMAGE_CLASS = "LoadImage"


class ComfyUIError(RuntimeError):
    """ComfyUI was reachable but could not produce a clip."""


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _upload_image(base_url: str, image_path: str, timeout: float) -> str:
    """POST an image to ComfyUI's /upload/image, return the name it stored
    it under -- that name, not our local path, is what a LoadImage node
    must reference.

    Hand-rolled multipart: this repo has no `requests`, and the frontend's
    own api.ts carries the same note about why a multipart body must not
    have its Content-Type set without the boundary.
    """
    path = Path(image_path)
    boundary = f"----claude{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "image/png"

    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ])

    req = urllib.request.Request(
        f"{base_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        stored = json.loads(resp.read().decode("utf-8"))
    name = stored.get("name")
    if not name:
        raise ComfyUIError(f"ComfyUI accepted the upload but returned no name: {stored}")
    subfolder = stored.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def _apply_prompt(workflow: dict, prompt: str) -> None:
    """Overwrite the positive CLIPTextEncode node's text.

    'Positive' is identified as the FIRST CLIPTextEncode in node-id order
    that is not obviously a negative prompt. ComfyUI workflows conventially
    carry two; picking by title when the exporter preserved one, else by
    order. Raises rather than silently rendering the default prompt -- a
    video generated from the workflow's leftover sample text is the kind
    of failure that looks like success until someone watches it.
    """
    candidates = [
        (node_id, node)
        for node_id, node in sorted(workflow.items())
        if node.get("class_type") == _POSITIVE_PROMPT_CLASS
    ]
    if not candidates:
        raise ComfyUIError(
            f"No {_POSITIVE_PROMPT_CLASS} node in the workflow -- cannot inject the prompt."
        )

    def _is_negative(node: dict) -> bool:
        title = (node.get("_meta", {}).get("title") or "").lower()
        return "negative" in title

    positive = next((n for _, n in candidates if not _is_negative(n)), candidates[0][1])
    positive.setdefault("inputs", {})["text"] = prompt


def _apply_reference_image(workflow: dict, stored_name: str) -> None:
    nodes = [n for _, n in sorted(workflow.items()) if n.get("class_type") == _LOAD_IMAGE_CLASS]
    if not nodes:
        raise ComfyUIError(
            f"reference_image was supplied but the workflow has no {_LOAD_IMAGE_CLASS} node "
            "-- export an image+text->video workflow for continuation scenes."
        )
    nodes[0].setdefault("inputs", {})["image"] = stored_name


def _await_history(base_url: str, prompt_id: str, timeout: float, poll: float) -> dict:
    """Poll /history/{id} until the run appears. ComfyUI returns {} for a
    job still in the queue, so an empty body is 'not done', not an error.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = _get_json(f"{base_url}/history/{prompt_id}", timeout=30)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise ComfyUIError(f"ComfyUI reported an error for {prompt_id}: {status}")
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(poll)
    raise ComfyUIError(
        f"ComfyUI did not finish prompt {prompt_id} within {timeout:.0f}s."
    )


def _first_video_output(entry: dict) -> dict:
    """Find the produced file in a history entry. Wan2.2 workflows emit
    under 'gifs' (the VideoHelperSuite node's key, used for mp4 too),
    'videos', or 'images' depending on the save node -- check all three
    rather than assuming the partner's exact node choice.
    """
    for node_output in entry.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            files = node_output.get(key)
            if files:
                return files[0]
    raise ComfyUIError(f"ComfyUI finished but produced no video output: {entry.get('outputs')}")


def _download(base_url: str, file_ref: dict, output_path: str, timeout: float) -> str:
    query = urllib.parse.urlencode({
        "filename": file_ref.get("filename", ""),
        "subfolder": file_ref.get("subfolder", ""),
        "type": file_ref.get("type", "output"),
    })
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{base_url}/view?{query}", timeout=timeout) as resp:
        out.write_bytes(resp.read())
    if out.stat().st_size == 0:
        raise ComfyUIError(f"Downloaded an empty file from ComfyUI for {file_ref}.")
    return str(out)


def make_renderer(
    base_url: str,
    workflow_path: str,
    *,
    job_timeout: float = 1800.0,
    poll_seconds: float = 3.0,
):
    """Build a generate_video(prompt, reference_image, duration, output_path)
    callable bound to this ComfyUI instance and workflow.

    Returned as a closure so it matches wan_client.generate_video's
    signature exactly and can be handed straight to the partner's
    run_phase_render(wan_generate_video=...).
    """
    base = base_url.rstrip("/")
    workflow_file = Path(workflow_path)
    if not workflow_file.exists():
        raise ComfyUIError(
            f"ComfyUI workflow not found at {workflow_file}. Export one from ComfyUI with "
            '"Save (API Format)" and point settings.comfyui_workflow_path at it.'
        )
    template = json.loads(workflow_file.read_text(encoding="utf-8"))
    client_id = uuid.uuid4().hex

    def generate_video(
        prompt: str,
        reference_image: Optional[str] = None,
        duration: int = 5,
        output_path: str = "output.mp4",
    ) -> str:
        workflow = json.loads(json.dumps(template))  # deep copy per scene
        _apply_prompt(workflow, prompt)
        if reference_image:
            _apply_reference_image(
                workflow, _upload_image(base, reference_image, timeout=120)
            )

        try:
            queued = _post_json(
                f"{base}/prompt",
                {"prompt": workflow, "client_id": client_id},
                timeout=60,
            )
        except urllib.error.URLError as exc:
            raise ComfyUIError(f"Cannot reach ComfyUI at {base}: {exc}") from exc

        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"ComfyUI did not queue the workflow: {queued}")

        entry = _await_history(base, prompt_id, timeout=job_timeout, poll=poll_seconds)
        return _download(base, _first_video_output(entry), output_path, timeout=600)

    return generate_video
