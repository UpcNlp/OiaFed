"""Tiny ImageNet dataset with the FedSRA reference transforms."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from ...registry import dataset


TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2770, 0.2691, 0.2821)


@dataset(
    name="tiny_imagenet",
    description="Tiny ImageNet 200-class image classification dataset",
    version="1.0",
    author="FedSRA",
    dataset_type="image_classification",
    num_classes=200,
    input_shape=(64, 64, 3),
)
class TinyImageNetDataset(Dataset):
    """Load the original or ImageFolder-reorganized Tiny ImageNet layout."""

    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        download: bool = False,
        augmentation: bool = True,
        transform_profile: str = "standard",
    ):
        del download  # The 237 MiB archive is prepared once on shared storage.
        requested_root = Path(data_dir)
        self.root = (
            requested_root
            if requested_root.name == "tiny-imagenet-200"
            else requested_root / "tiny-imagenet-200"
        )
        self.split = split
        self.transform_profile = transform_profile.lower()
        if self.transform_profile not in {"standard", "fedsra", "fafi", "oneshot_half"}:
            raise ValueError(
                "unsupported Tiny ImageNet transform_profile"
            )
        if not (self.root / "train").is_dir():
            raise FileNotFoundError(
                f"Tiny ImageNet is missing at {self.root}; expected train/ and val/"
            )

        is_train = split in {"train", "valid"}
        self.transform = self._build_transform(is_train and augmentation)
        train_probe = datasets.ImageFolder(self.root / "train")
        self.class_to_idx = train_probe.class_to_idx

        if is_train:
            self.dataset = datasets.ImageFolder(
                self.root / "train",
                transform=self.transform,
            )
            self.samples = self.dataset.samples
            self.targets = self.dataset.targets
        else:
            self.dataset = None
            self.samples = self._validation_samples()
            self.targets = [label for _, label in self.samples]

    def _build_transform(self, augmented: bool):
        if self.transform_profile == "fafi":
            return transforms.ToTensor()
        if self.transform_profile == "oneshot_half":
            operations = []
            if augmented:
                operations.extend([transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip()])
            operations.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            return transforms.Compose(operations)
        if augmented and self.transform_profile == "fedsra":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])
        if augmented:
            return transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ])

    def _validation_samples(self) -> list[tuple[Path, int]]:
        validation_root = self.root / "val"
        annotation_file = validation_root / "val_annotations.txt"
        samples: list[tuple[Path, int]] = []
        if annotation_file.is_file():
            for line in annotation_file.read_text(encoding="utf-8").splitlines():
                filename, wnid, *_ = line.split("\t")
                if wnid in self.class_to_idx:
                    samples.append(
                        (validation_root / "images" / filename, self.class_to_idx[wnid])
                    )
        else:
            for wnid, label in sorted(self.class_to_idx.items()):
                class_root = validation_root / wnid
                if class_root.is_dir():
                    samples.extend(
                        (path, label)
                        for path in sorted(class_root.rglob("*.JPEG"))
                    )
        if not samples:
            raise FileNotFoundError(f"No Tiny ImageNet validation images under {validation_root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if self.dataset is not None:
            return self.dataset[index]
        path, label = self.samples[index]
        with Image.open(path) as image:
            inputs = self.transform(image.convert("RGB"))
        return inputs, label
