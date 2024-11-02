"""
Modified from torchvision.models.segmentation.deeplabv3_resnet50
URL: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
"""

from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.segmentation import LRASPP
from torchvision.models._utils import IntermediateLayerGetter, _ovewrite_value_param
from typing import Optional, Any
from torchinfo import summary


def _lraspp_efficientnetb0(backbone: None, num_classes: int) -> LRASPP:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    # stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    stage_indices = [0] + [i for i, b in enumerate(backbone)] + [len(backbone) - 1]
    low_pos = stage_indices[-6]
    high_pos = stage_indices[-1]
    low_channels = backbone[low_pos][2].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return LRASPP(backbone, low_channels, high_channels, num_classes)

def lraspp_efficientnetb0(
    *,
    weights = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[EfficientNet_B0_Weights] = EfficientNet_B0_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> LRASPP:
    """Constructs a Lite R-ASPP Network model with a MobileNetV2 backbone from
    `Searching for MobileNetV2 <https://arxiv.org/abs/1905.02244>`_ paper.

    Args:
        weights (:class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.MobileNet_V2_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    weights_backbone = EfficientNet_B0_Weights.verify(weights_backbone)

    if weights is not None:
        print('Loading LRASPP segmentation pretrained weights...')
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        print('Loading custom backbone')
        num_classes = 21

    backbone = efficientnet_b0(weights=weights_backbone)
    model = _lraspp_efficientnetb0(backbone, num_classes)

    if weights is not None:
        print('Loading LRASPP segmentation pretrained weights...')
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


if __name__ == '__main__':
    model = lraspp_efficientnetb0()
    summary(model)