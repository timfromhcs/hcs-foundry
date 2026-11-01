from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import gateway

router = APIRouter()


def _state(request: Request):
    return request.app.state.foundry


def _error(exc: gateway.GatewayError) -> HTTPException:
    status = 401 if exc.code == "invalid_key" else 402 if exc.code == "billing_required" else 429 if exc.code == "rate_limited" else 400
    return HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})


@router.get("/gateway")
def gateway_status(request: Request):
    return gateway.public_status(_state(request).settings.state_dir)


class GatewayConfig(BaseModel):
    key: str = Field(min_length=8, max_length=400)
    model: str | None = Field(default=None, max_length=120)
    enabled: bool | None = None


@router.put("/gateway")
def gateway_configure(request: Request, body: GatewayConfig):
    state = _state(request)
    try:
        gateway.list_models(body.key, limit=1)
    except gateway.GatewayError as exc:
        raise _error(exc)
    config = gateway.set_config(state.settings.state_dir, key=body.key, model=body.model, enabled=body.enabled)
    state.activity("gateway.configure", "AI Gateway key configured", {"model": config["model"]})
    return gateway.public_status(state.settings.state_dir)


class GatewayToggle(BaseModel):
    enabled: bool


@router.post("/gateway/enable")
def gateway_enable(request: Request, body: GatewayToggle):
    state = _state(request)
    config = gateway.get_config(state.settings.state_dir)
    if not config["configured"]:
        raise HTTPException(status_code=400, detail={"code": "not_configured", "message": "Add a gateway key first."})
    gateway.set_config(state.settings.state_dir, key=config["key"], model=config["model"], enabled=body.enabled)
    state.activity("gateway.enable", f"AI Gateway {'enabled' if body.enabled else 'disabled'}")
    return gateway.public_status(state.settings.state_dir)


@router.delete("/gateway")
def gateway_clear(request: Request):
    state = _state(request)
    gateway.set_config(state.settings.state_dir, key=None)
    state.activity("gateway.clear", "AI Gateway key removed")
    return gateway.public_status(state.settings.state_dir)


@router.get("/gateway/models")
def gateway_models(request: Request):
    state = _state(request)
    config = gateway.get_config(state.settings.state_dir)
    if not config["configured"]:
        raise HTTPException(status_code=400, detail={"code": "not_configured", "message": "Add a gateway key first."})
    try:
        return {"models": gateway.list_models(config["key"], limit=80)}
    except gateway.GatewayError as exc:
        raise _error(exc)


@router.post("/gateway/test")
def gateway_test(request: Request):
    state = _state(request)
    config = gateway.get_config(state.settings.state_dir)
    if not config["configured"]:
        return {"ok": False, "code": "not_configured", "message": "No key configured."}
    try:
        models = gateway.list_models(config["key"], limit=1)
    except gateway.GatewayError as exc:
        raise _error(exc)
    return {"ok": True, "models_visible": len(models), "model": config["model"]}
