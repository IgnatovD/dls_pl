from torchvision import transforms


def get_transforms(stage: str, resize: int):

    if stage == 'fit':
        return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor()])

    elif stage == 'test':
        return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor()])