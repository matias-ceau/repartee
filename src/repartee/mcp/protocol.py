import json, struct, anyio
class Frame:
    @staticmethod
    async def send(stream, obj: dict):
        raw = json.dumps(obj).encode()
        await stream.send_all(struct.pack("!I", len(raw)) + raw)

    @staticmethod
    async def recv(stream):
        hdr = await stream.receive_exactly(4)
        length = struct.unpack("!I", hdr)[0]
        data = await stream.receive_exactly(length)
        return json.loads(data)
