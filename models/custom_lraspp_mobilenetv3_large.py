"""
Modified from torchvision.models.segmentation.deeplabv3_resnet50
URL: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
"""

from torchvision.models.mobilenet import MobileNetV3
from torchvision.models.segmentation import LRASPP, LRASPP_MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models._utils import IntermediateLayerGetter, _ovewrite_value_param
from typing import Optional, Any
from torchinfo import summary


def _lraspp_mobilenetv3(backbone: MobileNetV3, num_classes: int) -> LRASPP:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return LRASPP(backbone, low_channels, high_channels, num_classes)

def lraspp_mobilenet_v3_large(
    *,
    weights: Optional[LRASPP_MobileNet_V3_Large_Weights] = 'DEFAULT',
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> LRASPP:
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.

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
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    weights = LRASPP_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        print('Loading LRASPP segmentation pretrained weights...')
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _lraspp_mobilenetv3(backbone, num_classes)

    if weights is not None:
        print('Loading custom backbone')
        print('Loading LRASPP segmentation pretrained weights...')
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


if __name__ == '__main__':
    model = lraspp_mobilenet_v3_large()
    summary(model)