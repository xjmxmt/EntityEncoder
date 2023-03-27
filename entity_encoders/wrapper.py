import torch


class Wrapper:

    def train(self, X_train, y_train, save_path=None, epochs=5000, batch_size=4096, lr=1e-3) -> None:
        raise NotImplementedError

    def eval(self, X_val, y_val, batch_size=4096) -> float:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
