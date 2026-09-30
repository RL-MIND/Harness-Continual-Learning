def __getattr__(name):
    if name == "Odyssey":
        from .odyssey import Odyssey

        return Odyssey
    raise AttributeError(name)


__all__ = ["Odyssey"]
