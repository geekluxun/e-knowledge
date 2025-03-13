import logging

from fastapi import Depends, Query, APIRouter

from eknowledge.dto.user import UserInfoDto, parse_UserInfo
from eknowledge.services.rag.context import MyStorageContext
from eknowledge.services.rag.query.query_store import query_from_vector_store

logger = logging.getLogger(__name__)

rag_router = APIRouter()


class GlobalStorageContext:
    pass


@rag_router.get("/query/")
async def query(
        userInfo: UserInfoDto = Depends(parse_UserInfo), prompt: str = Query(None, title="Prompt", description="A prompt string")):
    logger.info("prompt:%s", prompt)
    metadata_filter = {
        "department": "qa",
    }
    logger.info(userInfo)

    response = query_from_vector_store(MyStorageContext.storageContext().vector_store, metadata_filter=metadata_filter, prompt=prompt)
    logger.info("response:%s", response)
    return response
