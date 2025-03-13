import json
from http.client import HTTPException
from typing import Optional

from fastapi.params import Form
from pydantic import BaseModel, field_validator


class UserInfoDto(BaseModel):
    user_id: str
    department: Optional[str] = None

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value):
        if not value.strip():
            raise ValueError("user_id cannot be empty")
        return value

    @field_validator("department")
    @classmethod
    def validate_department(cls, value):
        if not value.strip():
            raise ValueError("department cannot be empty")
        return value


def parse_UserInfo(userInfo: Optional[str] = Form(None)) -> UserInfoDto:
    """解析 JSON 并进行 Pydantic 校验"""
    if userInfo is None:
        raise HTTPException(status_code=400, detail="Missing 'userInfo' field in request")

    try:
        userInfo_dict = json.loads(userInfo)
        return UserInfoDto(**userInfo_dict)  # Pydantic 校验
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in 'userInfo'")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
