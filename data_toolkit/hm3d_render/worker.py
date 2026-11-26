"""Private local socket worker: separates CUDA Python from Habitat's Python ABI."""
import os
import pickle
import socket
import struct
import sys
import traceback


def send(sock, value):
    data = pickle.dumps(value, protocol=4)
    sock.sendall(struct.pack('!Q', len(data)) + data)


def receive(sock):
    def read(n):
        chunks = bytearray()
        while len(chunks) < n:
            chunk = sock.recv(n-len(chunks))
            if not chunk:
                raise EOFError('CUDA renderer socket closed')
            chunks.extend(chunk)
        return chunks
    size = struct.unpack('!Q', read(8))[0]
    if size > 256 * 1024 * 1024:
        raise ValueError('Oversized renderer message')
    return pickle.loads(read(size))


def main():
    from .mesh import MeshCudaRenderer
    sock = socket.socket(fileno=int(sys.argv[1]))
    renderer = None
    try:
        config = receive(sock)
        renderer = MeshCudaRenderer(config['scene'], gpu_device_id=config['gpu'],
                                    basis_workers=config.get('basis_workers', 4),
                                    atlas_max_side=config.get('atlas_max_side', 8192))
        send(sock, {'ok': True})
        while True:
            request = receive(sock)
            if request is None:
                break
            result = {name: renderer.render(**camera) for name, camera in request.items()}
            send(sock, {'ok': True, 'observations': result})
    except BaseException:
        try:
            send(sock, {'ok': False, 'error': traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if renderer is not None:
            renderer.close()
        sock.close()


if __name__ == '__main__':
    main()
