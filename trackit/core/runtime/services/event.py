from typing import Iterable
from . import EpochBasedTrainerEventListenerRegistry


def emit_start_event(event_listener_registry_list: Iterable[EpochBasedTrainerEventListenerRegistry]):
    for event_listener_registry in event_listener_registry_list:
        for listener in event_listener_registry.list_on_start_event_listener():
            listener()


def emit_stop_event(event_listener_registry_list: Iterable[EpochBasedTrainerEventListenerRegistry]):
    for event_listener_registry in event_listener_registry_list:
        for listener in event_listener_registry.list_on_stop_event_listener():
            listener()


def emit_epoch_begin_event(event_listener_registry_list: Iterable[EpochBasedTrainerEventListenerRegistry],
                           epoch: int, is_train: bool):
    for event_listener_registry in event_listener_registry_list:
        for listener in event_listener_registry.list_on_epoch_begin_event_listener():
            listener(epoch, is_train)


def emit_epoch_end_event(event_listener_registry_list: Iterable[EpochBasedTrainerEventListenerRegistry],
                         epoch: int, is_train: bool):
    for event_listener_registry in event_listener_registry_list:
        for listener in event_listener_registry.list_on_epoch_end_event_listener():
            listener(epoch, is_train)


def emit_iteration_begin_event(event_listener_registry_list: Iterable[EpochBasedTrainerEventListenerRegistry], is_train: bool):
    for event_listener_registry in event_listener_registry_list:
        for listener in event_listener_registry.list_on_iteration_begin_event_listener():
            listener(is_train)


def emit_iteration_end_event(event_listener_registry_list: Iterable[EpochBasedTrainerEventListenerRegistry], is_train: bool):
    for event_listener_registry in event_listener_registry_list:
        for listener in event_listener_registry.list_all_on_iteration_end_event_listener():
            listener(is_train)
