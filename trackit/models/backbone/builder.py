from datetime import timedelta
import torch

from trackit.core.runtime.global_constant import get_global_constant
from trackit.miscellanies.torch.distributed.barrier import torch_distributed_zero_first
from trackit.miscellanies.torch.dtype import set_default_dtype


def _build_backbone(backbone_config: dict, load_pretrained=True, torch_jit_trace_compatible=True,
                    device: torch.device = torch.device('cpu'), dtype: torch.dtype=torch.float32):
    load_pretrained = backbone_config.get('pretrained', load_pretrained)
    backbone_build_params = backbone_config.get('parameters', {})
    if backbone_config['type'] == 'swin_transformer':
        from .swint.swin_transformer import build_swin_transformer_backbone
        if 'embed_dim' in backbone_build_params:
            backbone_build_params['overwrite_embed_dim'] = backbone_build_params['embed_dim']
            del backbone_build_params['embed_dim']
        if torch_jit_trace_compatible:
            backbone_build_params['fused_window_process'] = False
        with torch.device(device), set_default_dtype(dtype):
            backbone = build_swin_transformer_backbone(load_pretrained=load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'timm':
        from huggingface_hub.utils import enable_progress_bars
        enable_progress_bars()
        if get_global_constant('TIMM_USE_OLD_CACHE', default=True):
            from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
            import huggingface_hub.constants
            import huggingface_hub.utils._http
            old_hf_hub_offline = huggingface_hub.constants.HF_HUB_OFFLINE
            if not old_hf_hub_offline:
                huggingface_hub.constants.HF_HUB_OFFLINE = True
                huggingface_hub.utils._http.reset_sessions()
            import timm
            try:
                with torch.device(device), set_default_dtype(dtype):
                    backbone = timm.create_model(pretrained=load_pretrained, **backbone_build_params)
            except (LocalEntryNotFoundError, OfflineModeIsEnabled):
                huggingface_hub.constants.HF_HUB_OFFLINE = False
                huggingface_hub.utils._http.reset_sessions()
                with torch.device(device), set_default_dtype(dtype):
                    backbone = timm.create_model(pretrained=load_pretrained, **backbone_build_params)
            finally:
                if old_hf_hub_offline != huggingface_hub.constants.HF_HUB_OFFLINE:
                    huggingface_hub.constants.HF_HUB_OFFLINE = old_hf_hub_offline
                    huggingface_hub.utils._http.reset_sessions()
        else:
            import timm
            with torch.device(device), set_default_dtype(dtype):
                backbone = timm.create_model(pretrained=load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'swin_transformer_v2':
        from .swint.swin_transformer_v2 import build_swin_v2
        with torch.device(device), set_default_dtype(dtype):
            backbone = build_swin_v2(load_pretrained=load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'ConvNeXt':
        from .convnext.builder import build_convnext_backbone
        with torch.device(device), set_default_dtype(dtype):
            backbone = build_convnext_backbone(load_pretrained=load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'DINOv2':
        from .dinov2.builder import build_dino_v2_backbone
        with torch.device(device), set_default_dtype(dtype):
            backbone = build_dino_v2_backbone(load_pretrained=load_pretrained, **backbone_build_params)
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone


def build_backbone(backbone_config: dict, load_pretrained=True, torch_jit_trace_compatible=False,
                   device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
    with torch_distributed_zero_first(on_local_master=not get_global_constant('on_shared_file_system'), timeout=timedelta(hours=5)):
        return _build_backbone(backbone_config, load_pretrained, torch_jit_trace_compatible, device, dtype)
