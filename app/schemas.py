"""REST contracts. Coordinates are normalized to [0,1]."""
from typing import Annotated
from pydantic import BaseModel, Field, field_validator

Coordinate = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[Coordinate, Coordinate]


class CameraCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    source: str = Field(min_length=1, max_length=2048)
    client_id: str | None = None
    zone: list[Point] = Field(default_factory=list, max_length=30)
    tripwire: list[Point] = Field(default_factory=list, max_length=2)
    enabled: bool = True

    @field_validator('source')
    @classmethod
    def valid_source(cls, v: str) -> str:
        v = v.strip()
        if v.isdecimal() and 0 <= int(v) <= 32:
            return v
        if v.startswith(('rtsp://', 'rtsps://', 'http://', 'https://')):
            return v
        if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return v
        raise ValueError('Use webcam index (0), RTSP URL, HTTP stream URL, or local video file')

    @field_validator('zone')
    @classmethod
    def zone_size(cls, v: list[Point]) -> list[Point]:
        if v and len(v) < 3:
            raise ValueError('Intrusion zone requires at least 3 vertices')
        return v

    @field_validator('tripwire')
    @classmethod
    def line_size(cls, v: list[Point]) -> list[Point]:
        if v and len(v) != 2:
            raise ValueError('Tripwire requires exactly 2 points')
        if len(v) == 2 and v[0] == v[1]:
            raise ValueError('Tripwire points must be distinct')
        return v


class CameraUpdate(BaseModel):
    source: str | None = Field(default=None, min_length=1, max_length=2048)
    client_id: str | None = None

    @field_validator('source')
    @classmethod
    def valid_update_source(cls, v):
        if v is None: return v
        v=v.strip()
        if v.isdecimal() and 0 <= int(v) <= 32: return v
        if v.startswith(('rtsp://','rtsps://','http://','https://')): return v
        if v.lower().endswith(('.mp4','.avi','.mov','.mkv')): return v
        raise ValueError('Use webcam index, RTSP/HTTP stream URL, or local video file')
    name: str | None = Field(default=None, min_length=2, max_length=80)
    zone: list[Point] | None = Field(default=None, max_length=30)
    tripwire: list[Point] | None = Field(default=None, max_length=2)
    enabled: bool | None = None

    @field_validator('zone')
    @classmethod
    def zone_size(cls, v):
        if v is not None and v and len(v) < 3:
            raise ValueError('Zone requires at least 3 points')
        return v

    @field_validator('tripwire')
    @classmethod
    def line_size(cls, v):
        if v is not None and v and len(v) != 2:
            raise ValueError('Tripwire requires exactly 2 points')
        if v is not None and len(v) == 2 and v[0] == v[1]:
            raise ValueError('Tripwire points must be distinct')
        return v


class CameraConfigUpdate(BaseModel):
    fire_enabled: bool | None = None
    violence_enabled: bool | None = None
    anomaly_enabled: bool | None = None
    object_enabled: bool | None = None
    telegram_enabled: bool | None = None
    tts_enabled: bool | None = None
    sensitivity: float | None = Field(default=None, ge=0.1, le=5.0)


class TestAlertsRequest(BaseModel):
    camera_id: str | None = None