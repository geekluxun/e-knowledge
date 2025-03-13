import logging
import os
from pathlib import Path
from typing import List

from fastapi import File, UploadFile, Depends, APIRouter

from eknowledge.dto.user import UserInfoDto, parse_UserInfo

# 指定文件保存的本地路径
SAVE_DIR = os.getcwd() + "/../temp/docs"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)  # 确保目录存在


logger = logging.getLogger(__name__)

document_router = APIRouter()


@document_router.post("/upload/")
async def upload_files(
        files: List[UploadFile] = File(...),
        userInfo: UserInfoDto = Depends(parse_UserInfo),
):
    saved_files = []
    for file in files:
        file_location = os.path.join(SAVE_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append({"filename": file.filename, "saved_path": file_location})
    logger(userInfo)
    return {
        "uploaded_files": saved_files
    }
