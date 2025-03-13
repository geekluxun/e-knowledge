import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from eknowledge.api.v1.endpoints.document import document_router
from eknowledge.api.v1.endpoints.rag import rag_router
from eknowledge.core.logging import init_logging
from eknowledge.services.rag import init_rag_settings

# log
init_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPII应用程序 生命周期事件"""
    logger("🚀初始化RAG上下文配置...")
    init_rag_settings()

    yield  # 这里会进入 FastAPI 运行阶段

    logger("🛑FastAPI应用程序结束...")


app = FastAPI(lifespan=lifespan)
app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(document_router, prefix="/document", tags=["document"])


# 全局异常处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_messages = [f"{err['loc'][1]}: {err['msg']}" for err in errors]
    return JSONResponse(
        status_code=400,
        content={"detail": error_messages},
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 应用程序开始启动...")
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
