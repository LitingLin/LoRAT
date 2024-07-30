from typing import Sequence, Mapping, Tuple, Optional
import numpy as np
from trackit.data.utils.data_source_matcher.builder import build_data_source_matcher
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize


_builtin_rules = (
    {
        'match': {
            'name': 'LaSOT',
        },
        'action': {
            'to_1_based_indexing': True,  # like in MATLAB
            'rasterize': False,
        }
    },{
        'match': {
            'name': 'VastTrack',
        },
        'action': {
            'to_1_based_indexing': True,
            'rasterize': False,
        }
    }, {
        'match': {
            'name_regex': 'UAV',
        },
        'action': {
            'rasterize': False,
            'name_prefix': 'uav_'
        }
    }, {
        'match': {
            'name': 'TempleColor-128',
        },
        'action': {
            'rasterize': False,
            'name_prefix': 'tpl_'
        }
    }, {
        'match': {
            'name_regex': 'NFS',
        },
        'action': {
            'rasterize': False,
            'name_prefix': 'nfs_'
        }
    }, {
        'match':
            { 'name_regex': '.*' }, # match any
        'action': {
            'rasterize': False
        }
    }
)


class ExternalToolkitCompatibilityHelper:
    def __init__(self, rules: Sequence[Mapping] = _builtin_rules):
        self._rules = {}
        for rule in rules:
            self._rules[build_data_source_matcher(rule['match'])] = rule['action']

    def adjust_for_pytracking(self, dataset_name: str, sequence_name: str, predicted_bboxes: np.ndarray):
        actions = self._get_action(dataset_name)
        if actions is None:
            return sequence_name, predicted_bboxes
        else:
            return self._do_adjustment(actions, predicted_bboxes, sequence_name)

    def adjust(self, dataset_name: str, predicted_bboxes: np.ndarray):
        actions = self._get_action(dataset_name)
        if actions is None:
            return predicted_bboxes
        else:
            return self._do_adjustment(actions, predicted_bboxes, None)

    def _get_action(self, dataset_name: str):
        for dataset_name_matcher, actions in self._rules.items():
            if dataset_name_matcher(dataset_name, ()):
                return actions

        return None

    @staticmethod
    def _do_adjustment(adjustment_actions: dict, predicted_bboxes: np.ndarray, sequence_name: Optional[str]):
        if adjustment_actions.get('to_1_based_indexing', False):
            predicted_bboxes = predicted_bboxes.copy()
            predicted_bboxes += 1
        if 'coordinate_adjustment' in adjustment_actions:
            coordinate_adjustment = adjustment_actions['coordinate_adjustment']
            predicted_bboxes = predicted_bboxes.copy()
            predicted_bboxes[:, 0] += coordinate_adjustment[0]
            predicted_bboxes[:, 1] += coordinate_adjustment[1]
            predicted_bboxes[:, 2] += coordinate_adjustment[2]
            predicted_bboxes[:, 3] += coordinate_adjustment[3]
        if adjustment_actions.get('rasterize', False):
            predicted_bboxes = bbox_rasterize(predicted_bboxes)
        if 'name_prefix' in adjustment_actions and sequence_name is not None:
            sequence_name = adjustment_actions['name_prefix'] + sequence_name
        if sequence_name is not None:
            return sequence_name, predicted_bboxes
        return predicted_bboxes
