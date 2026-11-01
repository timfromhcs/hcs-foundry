from fastapi import APIRouter

from . import data, gateway, hub, lab, models, system, v3

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system.router, tags=["system"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(hub.router, tags=["hub"])
api_router.include_router(gateway.router, tags=["gateway"])
api_router.include_router(data.router, tags=["data"])
api_router.include_router(lab.router, tags=["lab"])
api_router.include_router(v3.router, tags=["v3"])

__all__ = ["api_router"]
