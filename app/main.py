from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1.endpoints import workflow
from .core.config import settings
from .core.logger import logger

# 在应用启动时记录日志
logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Workflow based LLM system",
    version=settings.VERSION,
)

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(
    workflow.router, prefix=settings.API_V1_STR + "/workflow", tags=["workflow"]
)


@app.get("/")
async def root():
    logger.info("Access root endpoint")
    return {"message": "Welcome to Workflow LLM System"}
