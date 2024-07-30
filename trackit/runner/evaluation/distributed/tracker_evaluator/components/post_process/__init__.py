from typing import Any, Dict

class TrackerOutputPostProcess:
    def start(self):
        pass

    def stop(self):
        pass

    def __call__(self, output: Any) -> Dict[str, Any]:
        raise NotImplementedError()
