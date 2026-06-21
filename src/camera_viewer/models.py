from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    type: str
    default: Any
    label: str = ""
    description: str = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    options: list[str] = Field(default_factory=list)


class OperationSpec(BaseModel):
    label: str
    description: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)


class PipelineStep(BaseModel):
    id: str = Field(default_factory=lambda: f"step-{uuid4().hex[:8]}")
    enabled: bool = True
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineSettings(BaseModel):
    steps: list[PipelineStep] = Field(default_factory=list)


class VideoSettings(BaseModel):
    source: str = "0"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    jpeg_quality: int = 85


class ProcessorConfig(BaseModel):
    video: VideoSettings = Field(default_factory=VideoSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
