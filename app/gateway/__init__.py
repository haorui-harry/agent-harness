"""Gateway package for optional HTTP API surfaces."""

from app.gateway.app import app, build_gateway_app

__all__ = ["app", "build_gateway_app"]
