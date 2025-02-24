import time
import typing
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.utils.logger.logger import api_logger


class DbOperationsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, db_operations: DatabaseOperations):
        super().__init__(app)
        self.db_operations = db_operations

    async def dispatch(self, request: Request, call_next: typing.Callable):
        request.state.db_operations = self.db_operations  # Attach DB operations to request

        response = await call_next(request)

        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    allowed_api_keys: set
    api_key_header: str

    def __init__(self, app: FastAPI, allowed_api_keys: set, api_key_header: str):
        super().__init__(app)

        if isinstance(allowed_api_keys, str) is False:
            raise ValueError("allowed_api_keys must be a str")

        if isinstance(api_key_header, str) is False:
            raise ValueError("api_key_header must be a string")

        self.allowed_api_keys = set(allowed_api_keys.split(","))
        self.api_key_header = api_key_header

    async def dispatch(self, request: Request, call_next: typing.Callable):
        # Skip API key check for specific paths if needed
        if request.url.path in ["/api/health"]:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get(self.api_key_header)

        if not api_key:
            return JSONResponse(content={"detail": "Unauthorized"}, status_code=401)

        # Validate API key
        if api_key not in self.allowed_api_keys:
            return JSONResponse(content={"detail": "Unauthorized"}, status_code=403)

        return await call_next(request)


class LoggingAndErrorHandlingMiddleware(BaseHTTPMiddleware):
    api_key_header: str

    def __init__(self, app: FastAPI, api_key_header: str):
        super().__init__(app)

        if isinstance(api_key_header, str) is False:
            raise ValueError("api_key_header must be a string")

        self.api_key_header = api_key_header

    async def dispatch(self, request: Request, call_next: typing.Callable):
        start_time = time.time()

        context = {
            "trace_id": str(uuid.uuid4()),
            "request": {
                "method": request.method,
                "url": request.url.path,
                "client_ip": request.client.host,
                "client_key": f"{request.headers.get(self.api_key_header, '')[:4]}*",
            },
        }

        api_logger.add_context(context)

        api_logger.info("API request started")

        try:
            response = await call_next(request)

            elapsed_time_ms = round((time.time() - start_time) * 1000)
            api_logger.add_context(
                {"elapsed_time": elapsed_time_ms, "response_status": response.status_code}
            )

            logger_method = api_logger.info
            message = "API request completed"

            if response.status_code >= 500:
                logger_method = api_logger.error
                message = "API request errored"

            elif response.status_code >= 400:
                logger_method = api_logger.warning

            logger_method(message)

            return response
        except Exception:
            elapsed_time_ms = round((time.time() - start_time) * 1000)
            api_logger.add_context({"elapsed_time": elapsed_time_ms, "response_status": 500})

            api_logger.exception("API request errored")

            return JSONResponse(content={"detail": "Internal Server Error"}, status_code=500)
