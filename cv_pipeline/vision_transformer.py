import torch
from torchvision.transforms import v2
class VisionTransformer:
    def __init__(self):
        self.base_transform = v2.Compose([
            v2.ToImage()
            #v2.Resize((224,224)),
            #v2.CenterCrop(224),
        ])
        self.final_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.main_transform = self._get_main_transform()
        self.train_transform = self._get_train_transform()

    def _get_main_transform(self) -> v2.Compose:
        return v2.Compose(self.base_transform.transforms + self.final_transform.transforms)
    def _get_train_transform(self) -> v2.Compose:
        """
         Sets the training data transformation pipeline for the vision transformer model.
         Adding data augmentation techniques to improve model generalization
        """
        augmentation = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2)
        ])
        return v2.Compose(
            self.base_transform.transforms + 
            augmentation.transforms + 
            self.final_transform.transforms
            )
    