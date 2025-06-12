import anyio, itertools
from .protocol import Frame

class Client:
    _ids = itertools.count(1)

    def __init__(self, addr="tcp://127.0.0.1:55855"):
        self.addr = addr

    async def _connect(self):
        proto, host, port = self.addr.split("://")[0], *self.addr.split("://")[1].split(":")
        if proto != "tcp":
            raise ValueError("Only tcp:// supported")
        return await anyio.connect_tcp(host, int(port))

    async def call(self, method, *args, **kw):
        ident = next(self._ids)
        async with await self._connect() as stream:
            await Frame.send(stream, dict(id=ident, m=method,
                                          args=list(args), kw=kw))
            res = await Frame.recv(stream)
            if "e" in res:
                raise RuntimeError(res["e"])
            return res.get("r")
