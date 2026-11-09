import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torchvision.transforms import functional as F

IMG_SIZE = 256
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
PAD_FILL = tuple(round(value * 255) for value in NORM_MEAN)
CONSENSUS_TEMPERATURE = 0.3
DISAGREEMENT_TEMPERATURE = 0.7
TEMPERATURE_RULES = ("T0.3", "T0.5")


def resize_pad(image):
	width, height = image.size
	scale = IMG_SIZE / max(width, height)
	new_width = max(1, round(width * scale))
	new_height = max(1, round(height * scale))
	image = F.resize(image, [new_height, new_width], antialias=True)
	left = (IMG_SIZE - new_width) // 2
	top = (IMG_SIZE - new_height) // 2
	return F.pad(image, [left, top, IMG_SIZE - new_width - left, IMG_SIZE - new_height - top], fill=PAD_FILL)


class RandomAreaCrop:
	def __init__(self, scale):
		self.scale = scale

	def __call__(self, image):
		top, left, height, width = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=(0.75, 4 / 3))
		return F.crop(image, top, left, height, width)


def build_valid_transform():
	return transforms.Compose([
		transforms.Lambda(resize_pad),
		transforms.ToTensor(),
		transforms.Normalize(NORM_MEAN, NORM_STD),
	])


def combine_probabilities(member_probs, temperature_rule="T0.3"):
	assert (member_probs.ndim == 2) and (member_probs.shape[1] == 4)
	votes = member_probs >= 0.5
	unanimous = (votes == votes[:, :1]).all(dim=1, keepdim=True)
	middle_mean = member_probs.sort(dim=1).values[:, 1:3].mean(dim=1, keepdim=True)
	probs = torch.where(unanimous, member_probs.mean(dim=1, keepdim=True), middle_mean)
	if temperature_rule == "T0.3":
		temperatures = torch.where(unanimous, CONSENSUS_TEMPERATURE, DISAGREEMENT_TEMPERATURE)
	elif temperature_rule == "T0.5":
		temperatures = 0.5
	else:
		raise ValueError(f"알 수 없는 temperature 규칙: {temperature_rule}")
	return torch.sigmoid(torch.logit(probs) / temperatures)


def build_model(pretrained=False):
	weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
	net = models.efficientnet_b0(weights=weights)
	net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
	return net


class EfficientNetEnsemble(nn.Module):
	def __init__(self, state_dicts):
		super().__init__()
		assert len(state_dicts) == 4, "앙상블 가중치 4개가 필요합니다."
		self.models = nn.ModuleList()
		for state_dict in state_dicts:
			model = build_model()
			model.load_state_dict(state_dict)
			self.models.append(model)

	def member_probabilities(self, inputs):
		return torch.cat([torch.sigmoid(model(inputs)) for model in self.models], dim=1)

	def forward(self, inputs, temperature_rule="T0.3"):
		return combine_probabilities(self.member_probabilities(inputs), temperature_rule)
