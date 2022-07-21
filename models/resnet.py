"""
PyTorch implementation of ResNet [He et al., 2015],
adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
with modifications:
	- # input channels on first convolution = 1
	- ReGP + Narrow RF, as suggested by Niizumi et al., 2022 [BYOL for Audio: Exploring Pre-trained General-purpose Audio Representations]
	- ResNetC&D modifications [He et al., 2018, Bag of Tricks for Image Classification with Convolutional Neural Networks]
"""
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
	"""3x3 convolution with padding"""
	return nn.Conv2d(
		in_planes,
		out_planes,
		kernel_size=3,
		stride=stride,
		padding=dilation,
		groups=groups,
		bias=False,
		dilation=dilation,
	)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion: int = 1

	def __init__(
		self,
		inplanes: int,
		planes: int,
		stride: int = 1,
		downsample: Optional[nn.Module] = None,
		groups: int = 1,
		base_width: int = 64,
		dilation: int = 1,
		norm_layer: Optional[Callable[..., nn.Module]] = None,
	) -> None:
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError("BasicBlock only supports groups=1 and base_width=64")
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
	# while original implementation places the stride at the first 1x1 convolution(self.conv1)
	# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
	# This variant is also known as ResNet V1.5 and improves accuracy according to
	# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

	expansion: int = 4

	def __init__(
		self,
		inplanes: int,
		planes: int,
		stride: int = 1,
		downsample: Optional[nn.Module] = None,
		groups: int = 1,
		base_width: int = 64,
		dilation: int = 1,
		norm_layer: Optional[Callable[..., nn.Module]] = None,
	) -> None:
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.0)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(
		self,
		block: Type[Union[BasicBlock, Bottleneck]],
		layers: List[int],
		strides: List[int],
		ReGP: bool = False,
		C: bool = True,
		D: bool = True,
		num_classes: int = 1000,
		zero_init_residual: bool = False,
		groups: int = 1,
		width_per_group: int = 64,
		replace_stride_with_dilation: Optional[List[bool]] = None,
		norm_layer: Optional[Callable[..., nn.Module]] = None,
	) -> None:
		super().__init__()

		self.ReGP = ReGP
		self.D = D  # ResNetD modification

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError(
				"replace_stride_with_dilation should be None "
				f"or a 3-element tuple, got {replace_stride_with_dilation}"
			)
		self.groups = groups
		self.base_width = width_per_group
		if C:  # ResNetC modification
			self.conv1 = nn.Sequential(
				nn.Conv2d(1, self.inplanes//2, kernel_size=3, stride=strides[0], padding=1, bias=False), 
				norm_layer(self.inplanes//2),
				nn.ReLU(inplace=True),
				nn.Conv2d(self.inplanes//2, self.inplanes//2, kernel_size=3, stride=1, padding=1, bias=False),
				norm_layer(self.inplanes//2),
				nn.ReLU(inplace=True), 
				nn.Conv2d(self.inplanes//2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
				norm_layer(self.inplanes),
				nn.ReLU(inplace=True),
			)
		else:
			self.conv1 = nn.Sequential(
				nn.Conv2d(1, self.inplanes, kernel_size=7, stride=strides[0], padding=3, bias=False),
				norm_layer(self.inplanes),
				nn.ReLU(inplace=True),
			)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[1])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2], dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilate=replace_stride_with_dilation[2])
		if not ReGP:
			self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.LazyLinear(num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck) and m.bn3.weight is not None:
					nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
				elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
					nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

	def _make_layer(
		self,
		block: Type[Union[BasicBlock, Bottleneck]],
		planes: int,
		blocks: int,
		stride: int = 1,
		dilate: bool = False,
	) -> nn.Sequential:
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			if self.D:
				downsample = nn.Sequential(
					nn.AvgPool2d(stride, stride),
					conv1x1(self.inplanes, planes * block.expansion),
					norm_layer(planes * block.expansion),
				)
			else:
				downsample = nn.Sequential(
					conv1x1(self.inplanes, planes * block.expansion, stride),
					norm_layer(planes * block.expansion),
				)

		layers = []
		layers.append(
			block(
				self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
			)
		)
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(
				block(
					self.inplanes,
					planes,
					groups=self.groups,
					base_width=self.base_width,
					dilation=self.dilation,
					norm_layer=norm_layer,
				)
			)

		return nn.Sequential(*layers)

	def _forward_impl(self, x: Tensor) -> Tensor:
		# See note [TorchScript super()]
		x = self.conv1(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		if self.ReGP:
			x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
			B, T, D, C = x.shape
			x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
			(x1, _) = torch.max(x, dim=1)
			x2 = torch.mean(x, dim=1)
			x = x1 + x2
		else:
			x = self.avgpool(x)
			x = torch.flatten(x, 1)
		x = self.fc(x)

		return x

	def forward(self, x: Tensor) -> Tensor:
		return self._forward_impl(x)


def _resnet(
	block: Type[Union[BasicBlock, Bottleneck]],
	layers: List[int],
	**kwargs: Any,
) -> ResNet:

	model = ResNet(block, layers, [2, 1, 2, 2, 2], **kwargs)
	return model


def resnet18(**kwargs: Any) -> ResNet:
	"""ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
	Args:
		**kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
			base class. Please refer to the `source code
			<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
			for more details about this class.
	"""
	return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
	"""ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
	Args:
		**kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
			base class. Please refer to the `source code
			<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
			for more details about this class.
	"""
	return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
	"""ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
	.. note::
	   The bottleneck of TorchVision places the stride for downsampling to the second 3x3
	   convolution while the original paper places it to the first 1x1 convolution.
	   This variant improves the accuracy and is known as `ResNet V1.5
	   <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
	Args:
		**kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
			base class. Please refer to the `source code
			<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
			for more details about this class.
	"""
	return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


"""------------------------------Modified ResNets (ReGP + Narrow RF)-----------------------------"""



def _resnet_ReGP_NRF(
	block: Type[Union[BasicBlock, Bottleneck]],
	layers: List[int],
	**kwargs: Any,
) -> ResNet:

	model = ResNet(block, layers, [1, 1, 2, 2, [1, 2]], ReGP=True, **kwargs)
	return model


def resnet18_ReGP_NRF(**kwargs: Any) -> ResNet:
	return _resnet_ReGP_NRF(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_ReGP_NRF(**kwargs: Any) -> ResNet:
	return _resnet_ReGP_NRF(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_ReGP_NRF(**kwargs: Any) -> ResNet:
	return _resnet_ReGP_NRF(Bottleneck, [3, 4, 6, 3], **kwargs)



if __name__ == "__main__":

	model = resnet50()
	model.fc = nn.Identity()
	x = torch.randn((1, 1, 64, 96))
	out = model(x)
	print(out.shape)