def save_objects(objects, path: str, engine: str):
    if engine == 'torch':
        import torch
        return torch.save(objects, path)
    elif engine == 'json':
        import json
        with open(path, 'w', encoding='utf-8', newline='') as f:
            json.dump(objects, f)
    elif engine == 'pickle':
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(objects, f)
    else:
        raise ValueError(f'Unsupported engine: {engine}')


def load_objects(path: str, engine: str):
    if engine == 'torch':
        import torch
        return torch.load(path, map_location='cpu')
    elif engine == 'json':
        import json
        with open(path, 'r', encoding='utf-8', newline='') as f:
            return json.load(f)
    elif engine == 'pickle':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
