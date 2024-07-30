from __future__ import annotations
from dataclasses import dataclass, field
from . import EvaluationTaskTracer, EvaluationProgress, EvaluationProgress_Dataset
from typing import Sequence, Set, MutableMapping
from trackit.data.context.variable.eval import DatasetEvaluationTask


class EvaluationTaskTracer_Predefined(EvaluationTaskTracer):
    @dataclass
    class TrackTracingContext:
        repeat_index: int

    @dataclass
    class DatasetTracingContext:
        task_defined: DatasetEvaluationTask
        finished_tracks: Set[str] = field(default_factory=set)
        all_repeat_finished_tracks: Sequence[Set[str]] = field(init=False)
        tracing_tracks: MutableMapping[str, EvaluationTaskTracer_Predefined.TrackTracingContext] = field(default_factory=dict)

        def __post_init__(self):
            self.all_repeat_finished_tracks = [set() for _ in range(self.task_defined.repeat_times)]

    def __init__(self, predefined_evaluation_tasks: Sequence[DatasetEvaluationTask]):
        tracing_datasets = {}
        for predefined_task in predefined_evaluation_tasks:
            assert predefined_task.dataset_full_name not in tracing_datasets
            tracing_datasets[predefined_task.dataset_full_name] = EvaluationTaskTracer_Predefined.DatasetTracingContext(predefined_task)
        self._context = tracing_datasets

    def is_finished(self, dataset_full_name: str) -> bool:
        dataset_context = self._context[dataset_full_name]
        return len(dataset_context.finished_tracks) == len(dataset_context.task_defined.track_names)

    def submit(self, dataset_full_name: str, track_name: str) -> EvaluationProgress:
        dataset_context = self._context[dataset_full_name]
        assert track_name in dataset_context.task_defined.track_names
        assert track_name not in dataset_context.finished_tracks
        repeat_times = dataset_context.task_defined.repeat_times
        if track_name not in dataset_context.tracing_tracks:
            repeat_index = 0
            dataset_context.all_repeat_finished_tracks[repeat_index].add(track_name)
            if repeat_times > 1:
                dataset_context.tracing_tracks[track_name] = EvaluationTaskTracer_Predefined.TrackTracingContext(repeat_index)
            else:
                dataset_context.finished_tracks.add(track_name)
        else:
            tracing_track_context = dataset_context.tracing_tracks[track_name]
            repeat_index = tracing_track_context.repeat_index + 1
            assert repeat_index < repeat_times
            dataset_context.all_repeat_finished_tracks[repeat_index].add(track_name)
            if repeat_index + 1 == repeat_times:
                del dataset_context.tracing_tracks[track_name]
                dataset_context.finished_tracks.add(track_name)
            else:
                tracing_track_context.repeat_index = repeat_index
        total_num_tracks = len(dataset_context.task_defined.track_names)
        all_evaluated = len(dataset_context.finished_tracks) == total_num_tracks
        this_repeat_all_evaluated = len(dataset_context.all_repeat_finished_tracks[repeat_index]) == total_num_tracks
        return EvaluationProgress(repeat_index, EvaluationProgress_Dataset(repeat_times, all_evaluated, this_repeat_all_evaluated))
