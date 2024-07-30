from trackit.core.runtime.build_context import BuildContext
from trackit.data.utils.data_source_matcher.builder import build_data_source_matcher
from .collector import EvaluationResultCollector_RuntimeIntegration, SubHandlerBuildOptions


def build_evaluation_result_collector(result_collector_config: dict, build_context: BuildContext):
    dispatch_config = result_collector_config['dispatch']
    handler_build_options = {}
    for sub_collector_config in dispatch_config:
        data_source_matcher = build_data_source_matcher(sub_collector_config['match'])
        handler_configs = sub_collector_config['handlers']
        build_options = []
        for handler_config in handler_configs:
            build_option = SubHandlerBuildOptions(handler_config['type'],
                                                  handler_config.get('file_name', None),
                                                  handler_config.get('bbox_rasterize', False))
            build_options.append(build_option)
        handler_build_options[data_source_matcher] = build_options

    optional_predefined_evaluation_tasks = None
    if 'evaluation_task_desc' in build_context.variables:
        optional_predefined_evaluation_tasks = build_context.variables['evaluation_task_desc']

    run_async = result_collector_config['async_worker']
    log_summary = result_collector_config['log_summary']

    result_collector = EvaluationResultCollector_RuntimeIntegration(build_context.name,
                                                                    handler_build_options,
                                                                    optional_predefined_evaluation_tasks,
                                                                    run_async,
                                                                    log_summary)
    build_context.services.event.register_on_epoch_begin_event_listener(lambda epoch, is_train: result_collector.on_epoch_begin())
    build_context.services.event.register_on_epoch_end_event_listener(lambda epoch, is_train: result_collector.on_epoch_end())
    build_context.services.batch_collective_communication.all_gather.register(result_collector.distributed_prepare_gathering, result_collector.distributed_on_gathered)
    return result_collector
