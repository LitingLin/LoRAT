import sys
from typing import Optional
import os
import zmq
from zmq import Context
import multiprocessing
import pickle
import psutil


def get_listening_ports(with_process_name: bool = True):
    # Get a list of all active connections
    connections = psutil.net_connections(kind='inet')

    listening_ports = []

    for conn in connections:
        # Check if the connection status is 'LISTEN'
        if conn.status == psutil.CONN_LISTEN:
            if with_process_name:
                try:
                    # Fetch the process name using the PID
                    process = psutil.Process(conn.pid)
                    process_name = process.name()
                except psutil.NoSuchProcess:
                    process_name = "Unknown"
                listening_ports.append((process_name, conn.pid, conn.laddr))
            else:
                listening_ports.append((conn.pid, conn.laddr))

    return listening_ports


def print_all_listening_ports(with_process_name: bool = True):
    listening_ports = get_listening_ports(with_process_name)

    if with_process_name:
        string = "Process Name\tPID\tAddress\t\tPort"
        string += "\n-----------------------------------------"
    else:
        string = "PID\tAddress\t\tPort"
        string += "\n-----------------------"

    for record in listening_ports:
        if with_process_name:
            process_name, pid, laddr = record
            string += f"\n{process_name}\t{pid}\t{laddr.ip}\t\t{laddr.port}"
        else:
            pid, laddr = record
            string += f"\n{pid}\t{laddr.ip}\t\t{laddr.port}"
    print(string, flush=True)


class CallbackFactory:
    def __init__(self, callback_class, args=()):
        self.callback_class = callback_class
        self.args = args

    def __call__(self):
        return self.callback_class(*self.args)


class Response:
    def __init__(self):
        self.status_code = 200
        self.body = ''

    def set_status_code(self, code):
        self.status_code = code

    def set_body(self, body):
        self.body = body


def _event_loop(socket_address, key, callback, parent_pid: Optional[int]):
    interrupt = False
    if sys.platform == 'win32':
        from zmq.utils.win32 import allow_interrupt

        def _interrupt_handler():
            nonlocal interrupt
            interrupt = True

        interrupt_context = allow_interrupt(_interrupt_handler)
    else:
        from contextlib import nullcontext
        interrupt_context = nullcontext()
    default_timeout_ms = 1000
    ctx = Context()
    with ctx, interrupt_context:
        socket = ctx.socket(zmq.REP)
        if parent_pid is not None:
            socket.setsockopt(zmq.RCVTIMEO, default_timeout_ms)
        with socket:
            try:
                socket.bind(socket_address)
            except zmq.error.ZMQError as e:
                print(f'Failed to bind socket address {socket_address}. Error: {e}')
                print_all_listening_ports()
                raise e

            while True:
                try:
                    raw_command = socket.recv(copy=False)
                except zmq.error.Again:
                    if parent_pid is not None:
                        if not psutil.pid_exists(parent_pid):
                            break
                    if interrupt:
                        break
                    continue
                try:
                    command = pickle.loads(raw_command.bytes)
                except Exception as e:
                    print(f'Failed to unpickle command. Error: {e}')
                    continue

                if not isinstance(command, tuple) and len(command) != 2:
                    print(f'Invalid command: {command}')
                    socket.send(pickle.dumps((400,)), copy=False)
                    continue

                if command[0] != key:
                    socket.send(pickle.dumps((403,)), copy=False)
                    continue

                command = command[1]

                if command == 'hello':
                    socket.send(pickle.dumps((200,)), copy=False)
                elif command == 'shutdown':
                    break
                else:
                    response = Response()
                    callback(command, response)
                    socket.send(pickle.dumps((response.status_code, response.body)), copy=False)


def _server_entry(socket_address: str, key: str, callback, parent_pid: Optional[int] = None):
    if isinstance(callback, CallbackFactory):
        callback = callback()

    _event_loop(socket_address, key, callback, parent_pid)


class ServerLauncher:
    def __init__(self, socket_address: str, key: str, callback):
        self.socket_address = socket_address
        self.key = key
        self.callback = callback
        self.process = None
        self.stopped = None

    @staticmethod
    def try_bind_address(socket_address):
        try:
            ctx = Context()
            with ctx:
                socket = ctx.socket(zmq.REP)
                with socket:
                    socket.bind(socket_address)
            return True
        except Exception:
            return False

    def __del__(self):
        self.stop()

    def is_launched(self):
        return self.stopped is False

    def launch(self, wait_for_ready=True):
        assert self.stopped is not False
        self.process = multiprocessing.Process(target=_server_entry, args=(self.socket_address, self.key, self.callback, os.getpid()))
        self.process.start()
        if wait_for_ready:
            ctx = Context()
            with ctx:
                socket = ctx.socket(zmq.REQ)
                with socket:
                    socket.connect(self.socket_address)
                    socket.send_pyobj((self.key, 'hello'))
                    recv = socket.recv_pyobj()
            if not isinstance(recv, tuple) or recv[0] != 200:
                self.process.kill()
                raise Exception('Unexpected value')
        self.stopped = False

    def stop(self, wait_for_stop=True, waiting_timeout=5):
        if self.stopped is False:
            ctx = Context()
            with ctx:
                socket = ctx.socket(zmq.REQ)
                with socket:
                    socket.connect(self.socket_address)
                    socket.send_pyobj((self.key, 'shutdown'))
            if wait_for_stop:
                self.process.join(waiting_timeout)
                if self.process.exitcode is None:
                    self.process.kill()
                    print('Timeout when waiting for server process to exit. Killed.')
                self.process.close()
                del self.process
            self.stopped = True


class Client:
    def __init__(self, socket_address: str, key: str):
        self.socket_address = socket_address
        self.key = key

    def _initialize(self):
        if not hasattr(self, 'socket'):
            self.ctx = zmq.Context()
            self.socket = self.ctx.socket(zmq.REQ)
            self.socket.connect(self.socket_address)

    def start(self):
        self._initialize()

    def stop(self):
        if hasattr(self, 'socket'):
            try:
                self.socket.close()
            finally:
                self.ctx.term()
            del self.socket
            del self.ctx

    def __call__(self, *args):
        self._initialize()
        self.socket.send_pyobj((self.key, args))
        response = self.socket.recv_pyobj()
        if not isinstance(response, tuple):
            raise RuntimeError('unexpected response')
        if response[0] < 200 or response[0] >= 300:
            raise RuntimeError(f'remote procedure failed with code {response[0]}')
        else:
            return response[1]
