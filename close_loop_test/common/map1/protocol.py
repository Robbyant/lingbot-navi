"""Private parent/child pipe protocol; arrays never use temporary files.

Pickle is restricted to the process we launch, never a network listener.
"""
import pickle
import struct

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def send(stream, message):
    data = pickle.dumps(message, protocol=4)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError('online memory message exceeds 64 MiB')
    stream.write(struct.pack('!Q', len(data)))
    stream.write(data)
    stream.flush()


def _read(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        data = stream.read(size-len(chunks))
        if not data:
            raise EOFError('online memory pipe closed')
        chunks.extend(data)
    return bytes(chunks)


def receive(stream):
    size = struct.unpack('!Q', _read(stream, 8))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError('online memory message exceeds 64 MiB')
    return pickle.loads(_read(stream, size))
