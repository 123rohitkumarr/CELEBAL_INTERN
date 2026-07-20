from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routes.assistant import router


app = FastAPI(
title="Single Agent Smart Assistant",
description="A simple smart assistant built with FastAPI"
)

app.include_router(router)




@app.get("/")
async def root():
    return {"message": "Welcome to the Single Agent Smart Assistant!"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc)
        },
    )