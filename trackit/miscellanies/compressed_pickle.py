import pickle


def write_pkl(data, file, compress=True):
    if compress:
        import zstd
        with open(file, 'wb') as f:
            cctx = zstd.ZstdCompressor()
            with cctx.stream_writer(f) as compressor:
                pickle.dump(data, compressor)
    else:
        with open(file, 'wb') as f:
            pickle.dump(data, f)


def read_pkl(file, compress=True):
    if compress:
        import zstd
        with open(file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as decompressor:
                return pickle.load(decompressor)
    else:
        with open(file, 'rb') as f:
            return pickle.load(f)
