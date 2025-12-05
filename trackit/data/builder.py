from . import DataPipeline
from trackit.core.runtime.build_context import BuildContext
from trackit.miscellanies.printing import print_centered_text
from trackit.miscellanies.torch.config_parsing.dtype import get_torch_dtype_with_machine_compatibility_checking


def build_data_pipeline(name: str, data_config: dict, build_context: BuildContext, config: dict) -> DataPipeline:
    print_centered_text(f"Building data pipeline: {name}")
    data_pipeline_type = data_config['type']
    print('type:', data_pipeline_type)
    dtype = get_torch_dtype_with_machine_compatibility_checking(data_config.get('dtype', 'float32'),
                                                                build_context.device.type)
    print('dtype:', dtype)
    build_context.variables['dtype'] = dtype
    if data_pipeline_type == 'siamese_tracker_train':
        from .methods.siamese_tracker_train.builder import build_siamese_tracker_train_data_pipeline
        data_pipeline = build_siamese_tracker_train_data_pipeline(data_config, build_context, config, dtype)
    elif data_pipeline_type == 'siamese_tracker_eval':
        from .methods.siamese_tracker_eval.builder import build_siamese_tracker_eval_data_pipeline
        data_pipeline = build_siamese_tracker_eval_data_pipeline(data_config, build_context, config, dtype)
    elif data_pipeline_type == 'siamese_tracker_eval_vot':
        from .methods.siamese_tracker_eval.builder import build_siamese_tracker_eval_vot_integrated_data_pipeline
        data_pipeline = build_siamese_tracker_eval_vot_integrated_data_pipeline(data_config, build_context, config, dtype)
    elif data_pipeline_type == 'SPMTrack_train':
        from .methods.SPMTrack_train.builder import build_SPMTrack_train_data_pipeline
        data_pipeline = build_SPMTrack_train_data_pipeline(data_config, build_context, config, dtype)
    elif data_pipeline_type == 'LoRATv2_train':
        from .methods.LoRATv2_train.builder import build_LoRATv2_train_data_pipeline
        data_pipeline = build_LoRATv2_train_data_pipeline(data_config, build_context, config, dtype)
    else:
        raise NotImplementedError(f"Data pipeline type {data_pipeline_type} is not implemented yet.")
    print_centered_text('')
    return data_pipeline
