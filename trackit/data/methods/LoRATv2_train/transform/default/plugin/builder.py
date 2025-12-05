from dataclasses import dataclass
from typing import Optional
import torch
from trackit.core.runtime.build_context import BuildContext
from trackit.data import MainProcessDataPipeline
from . import ExtraTransform, ExtraTransform_DataCollector


def build_plugins(transform_config: dict, config: dict, build_context: BuildContext,
                  dtype: torch.dtype):
    register = PluginRegistrationHelper()
    if 'plugin' in transform_config:
        for plugin_config in transform_config['plugin']:
            if plugin_config['type'] == 'LoRATv1_label_generation':
                from .LoRATv1_label_gen.builder import build_LoRATv1_label_generator
                register.register(*build_LoRATv1_label_generator(plugin_config, config))
            elif plugin_config['type'] == 'LoRATv1_template_foreground_indicating_mask_generation':
                from .LoRATv1_template_foreground_indicating_mask_gen.builder import (
                    build_LoRATv1_template_feat_foreground_mask_generator)
                register.register(*build_LoRATv1_template_feat_foreground_mask_generator(plugin_config, config))
            elif plugin_config['type'] == 'add_param':
                from .add_param import build_add_param_data_pipeline
                register.register(*build_add_param_data_pipeline(plugin_config))
    return register.extra_transforms, register.extra_data_collators, register.extra_data_pipelines_on_main_process


@dataclass(frozen=True)
class PluginRegistrationHelper:
    extra_transforms = []
    extra_data_collators = []
    extra_data_pipelines_on_main_process = []

    def register(self, transform: Optional[ExtraTransform] = None,
                 data_collator: Optional[ExtraTransform_DataCollector] = None,
                 data_pipeline_on_main_process: Optional[MainProcessDataPipeline] = None):
        if transform is not None:
            self.extra_transforms.append(transform)
        if data_collator is not None:
            self.extra_data_collators.append(data_collator)
        if data_pipeline_on_main_process is not None:
            self.extra_data_pipelines_on_main_process.append(data_pipeline_on_main_process)
