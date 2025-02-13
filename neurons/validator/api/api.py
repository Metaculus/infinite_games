from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from uvicorn import Config, Server

from neurons.validator.api.middlewares import (
    APIKeyMiddleware,
    DbOperationsMiddleware,
    LoggingAndErrorHandlingMiddleware,
)
from neurons.validator.api.routes.root_router import router as root_router
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.utils.logger.logger import set_uvicorn_logger


class API:
    host: str
    port: int
    fast_api: FastAPI | None
    server: Server | None
    db_operations: DatabaseOperations

    def __init__(
        self, host: str, port: int, db_operations: DatabaseOperations, api_access_key: str | None
    ):
        if not isinstance(host, str):
            raise ValueError("host must be a string")

        if not isinstance(port, int):
            raise ValueError("port must be an integer")

        if not isinstance(db_operations, DatabaseOperations):
            raise ValueError("db_operations must be an instance of DatabaseOperations")

        if api_access_key is not None and not isinstance(api_access_key, str):
            raise ValueError("api_access_key must be a string or None")

        self.host = host
        self.port = port
        self.fast_api = None
        self.server = None
        self.db_operations = db_operations
        self.api_access_key = api_access_key
        self.api_key_header = "X-API-Key"

    def _get_db_operations(self):
        return self.db_operations

    def _get_rate_limiter_key(self, request: Request):
        return request.headers.get(self.api_key_header, "fallback-key")

    def create_api(self):
        # Create FastAPI
        fast_api = FastAPI(dependencies=[Depends(self._get_db_operations)])

        # Rate limiter
        limiter = Limiter(key_func=self._get_rate_limiter_key, application_limits=["1/1seconds"])
        fast_api.state.limiter = limiter

        fast_api.add_middleware(DbOperationsMiddleware, db_operations=self.db_operations)

        fast_api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        fast_api.add_middleware(SlowAPIMiddleware)

        fast_api.add_middleware(
            APIKeyMiddleware,
            allowed_api_key=self.api_access_key,
            api_key_header=self.api_key_header,
        )

        fast_api.add_middleware(LoggingAndErrorHandlingMiddleware)

        fast_api.include_router(root_router, prefix="/api")

        return fast_api

    async def start(self):
        if not self.api_access_key:
            return

        if self.fast_api is not None:
            raise RuntimeError("API already started")

        fast_api = self.create_api()

        config = Config(
            fast_api, host=self.host, port=self.port, access_log=False, use_colors=False
        )

        set_uvicorn_logger()

        self.fast_api = fast_api
        self.server = Server(config)

        await self.server.serve()
