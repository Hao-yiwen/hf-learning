import importlib.util, torch
print("torch =", torch.__version__)
spec = importlib.util.find_spec("torchvision")
print("torchvision spec =", spec.origin if spec else None)