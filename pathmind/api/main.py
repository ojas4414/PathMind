import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from torch.distributions import Categorical
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "build" / "Release"))
sys.path.insert(0, str(BASE / "python"))

import traffic_env  # noqa: E402  (pybind11 module)
from agent import PolicyNetwork  # noqa: E402

policy = PolicyNetwork()
policy.load_state_dict(
    torch.load(BASE / "trained_policy.pth", map_location="cpu", weights_only=True)
)
policy.eval()

clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None


def run_episode_loop() -> None:
    env = traffic_env.trafficEnv(200)
    obs = env.reset()
    total_reward = 0.0
    step = 0

    while True:
        tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            probs = policy(tensor)
        action = Categorical(probs).sample().item()

        obs, reward, done = env.step(action)
        total_reward += reward
        step += 1

        # obs layout: signal_states[0:10], car_volume[10:20], occupancy[20:30]
        payload = json.dumps(
            {
                "step": step,
                "action": action,
                "car_volume": [round(v, 2) for v in obs[10:20]],
                "signal_states": [round(v, 2) for v in obs[0:10]],
                "reward": round(reward, 2),
                "total_reward": round(total_reward, 2),
            }
        )

        if _loop is not None and clients:
            asyncio.run_coroutine_threadsafe(_broadcast(payload), _loop)

        if done:
            obs = env.reset()
            total_reward = 0.0
            step = 0

        time.sleep(0.5)


async def _broadcast(message: str) -> None:
    dead: set[WebSocket] = set()
    for ws in list(clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    threading.Thread(target=run_episode_loop, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        clients.discard(websocket)
