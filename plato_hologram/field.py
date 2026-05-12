"""Holographic knowledge field — tile=embedding, every tile encodes the field."""
import struct, hashlib, math

TILE_BYTES = 64; EMBED_DIMS = 8

def tile_to_embedding(tile_bytes):
    h = hashlib.sha256(tile_bytes).digest()
    return [(struct.unpack('H', h[i*2:(i+1)*2])[0] / 65535.0) * 2.0 - 1.0 for i in range(EMBED_DIMS)]

class HologramField:
    def __init__(self):
        self.tiles = {}; self.centroid = [0.0]*EMBED_DIMS
        self.boundary = [0.0]*EMBED_DIMS; self.n = 0
    
    def add(self, data, meta=""):
        tb = data.encode()[:TILE_BYTES].ljust(TILE_BYTES, b'\x00') if isinstance(data, str) else data
        emb = tile_to_embedding(tb)
        h = hashlib.md5(tb).hexdigest()[:16]
        self.tiles[h] = {"emb": emb, "meta": meta, "n": self.n}; self.n += 1
        for i in range(EMBED_DIMS):
            self.centroid[i] = (self.centroid[i]*(self.n-1)+emb[i])/self.n
            self.boundary[i] = max(self.boundary[i], abs(emb[i]-self.centroid[i]))
        return h
    
    def nearest(self, query, k=5):
        q = tile_to_embedding(query.encode()[:TILE_BYTES].ljust(TILE_BYTES, b'\x00')) if isinstance(query, str) else query
        dists = [(math.sqrt(sum((q[j]-t["emb"][j])**2 for j in range(EMBED_DIMS))), h, t) for h, t in self.tiles.items()]
        return sorted(dists)[:k]
    
    def density(self, point):
        nn = self.nearest(point, k=min(10, len(self.tiles)))
        return 1.0/(1.0+sum(d for d,*_ in nn)/len(nn)) if nn else 0.0
    
    def onboard(self, n=5):
        return [{"embedding": t["emb"], "centroid": self.centroid, "neighbors": len(self.tiles)}
                for t in [self.tiles[h] for h in sorted(self.tiles.keys())[:n]]]
