"""
Caller Role Seam
────────────────
Who is asking, and are they allowed to do this. Third member of the same
family as app/config.py's get_tenant_id (which tenant's data) and
get_user_id (which user) -- and it carries the same warning as both:

    THIS IS A SEAM, NOT SECURITY.

get_role() reads a client-supplied `X-User-Role` header. A caller can put
whatever it likes there, exactly as a caller could once put whatever it
liked in `tenant_id` (the recorded bug at quiz.py's original
`request.tenant_id or "company_abc"`). The difference is that get_tenant_id
resolved that by ignoring client input entirely, which works when there is
one right answer per process; role has no such fixed answer, so the header
is read but must never be mistaken for proof.

The point of the seam is shape: every call site already asks "what is this
caller allowed to do", so replacing this body with a validated JWT claim
later changes this file and nothing else. Until that exists, the real
enforcement for an unattended deployment is the same as everywhere else in
this codebase -- do not expose the port (see settings.uploads_read_only's
comment for the same reasoning applied to uploads).

Introduced 2026-09-21 for video generation: course-creation features
(video) are Admin/Tenant-only, while a tenant's employees get the tutor
(chat, quiz, voice) and nothing that writes course content.
"""
from enum import Enum
from typing import Optional

from fastapi import Header

from app.config import get_settings
from app.errors import ForbiddenError


class Role(str, Enum):
    """Who is calling.

    ADMIN    -- platform operator, across tenants
    TENANT   -- the customer company's own administrator: uploads course
                documents, creates courses and their videos
    EMPLOYEE -- that company's learner: consumes the tutor, creates nothing
    """

    ADMIN = "admin"
    TENANT = "tenant"
    EMPLOYEE = "employee"


# Course-creation permission. Video generation is part of authoring a
# course, not of taking one, so an employee must not reach it -- the
# 2026-09-21 requirement correction (the earlier assumption was that any
# tenant user could generate videos).
COURSE_AUTHOR_ROLES = frozenset({Role.ADMIN, Role.TENANT})


def resolve_role(raw: Optional[str]) -> Role:
    """Parse a role name, falling back to settings.default_user_role.

    Unknown values fall back rather than 400 deliberately: a typo'd or
    stale header from an older client should degrade to the configured
    default (and then be refused by require_role if that default is not
    permitted), not fail the request with a shape error that tells a
    probing caller exactly which role names exist.

    Split out from get_role so it is callable without FastAPI's dependency
    machinery -- this suite calls router functions directly (see
    tests/test_ingest_router.py's header), and the worker/scripts have no
    request object at all.
    """
    for candidate in (raw, get_settings().default_user_role):
        if candidate is None:
            continue
        try:
            return Role(str(candidate).strip().lower())
        except ValueError:
            continue
    return Role.EMPLOYEE  # least-privileged last resort


def get_role(x_user_role: Optional[str] = Header(None, alias="X-User-Role")) -> Role:
    """FastAPI dependency: the calling role for this request."""
    return resolve_role(x_user_role)


def require_role(role: Role, allowed: frozenset, action: str) -> None:
    """Raise ForbiddenError unless `role` is in `allowed`.

    A plain function rather than a dependency factory so a router can call
    it after its own validation and so tests can exercise it by calling the
    endpoint function directly with role=..., no HTTP layer needed.
    """
    if role not in allowed:
        raise ForbiddenError(
            action=action,
            role=role.value,
            allowed=", ".join(sorted(r.value for r in allowed)),
        )
