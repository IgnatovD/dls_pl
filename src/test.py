from src.train_utils import load_object

a = load_object('torch.nn.BCEWithLogitsLoss')(**{})
print(type(a))