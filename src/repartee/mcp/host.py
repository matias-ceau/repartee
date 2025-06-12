import anyio
from .protocol import Frame

class Host:
    def __init__(self):  # map method-name → coroutine
        self._handlers = {}

    def on(self, name):            # decorator
        def wrap(fn):
            self._handlers[name] = fn
            return fn
        return wrap

    async def _serve_stream(self, stream):
        async with stream:
            while True:
                try:
                    req = await Frame.recv(stream)
                except BaseException:
                    break
                fn = self._handlers.get(req["m"])
                reply = {"id": req.get("id")}
                try:
                    reply["r"] = await fn(*req.get("args", []),
                                          **req.get("kw", {}))
                except Exception as exc:
                    reply["e"] = repr(exc)
                await Frame.send(stream, reply)

    async def serve(self, addr="tcp://127.0.0.1:55855"):
        proto, host, port = addr.split("://")[0], *addr.split("://")[1].split(":")
        port = int(port)
        if proto != "tcp":
            raise ValueError("Only tcp:// supported")
        await anyio.create_tcp_listener(local_host=host,
                                        local_port=port,
                                        task_group=lambda tg:
                                            tg.start_soon(self._accept_loop))

    async def _accept_loop(self, task_status=anyio.TASK_STATUS_IGNORED):
        task_status.started()
        async with anyio.create_tcp_listener(local_host="127.0.0.1",
                                             local_port=55855) as listener:
            async for stream in listener:
                await anyio.create_task_group().start(self._serve_stream, stream)
