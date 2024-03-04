"""Classes for executing basic commands either locally or remotely. Useful for
dependency injection."""

from abc import ABC, abstractmethod
from fabric import Connection
from typing import Tuple

class Executor(ABC):
    # enforce the existence of a working directory property
    @property
    @abstractmethod
    def working_dir(self) -> str:
        """Working directory of the executor.
        
        Returns:
            str: Working directory of the executor.
        """
        pass

    @working_dir.setter
    @abstractmethod
    def working_dir(self, working_dir: str) -> None:
        """Set the working directory of the executor.

        Args:
            working_dir (str): Working directory of the executor.
        """
        pass
    
    @abstractmethod
    def run(self, cmd: str) -> Tuple[str, str]:
        """Run a command on the executor, return its stdout and stderr.

        Args:
            cmd (str): Command to run
        
        Returns:
            Tuple[str, str]: stdout and stderr of the command.
        """
        pass

    @abstractmethod
    def put(self, local_path: str, remote_path: str) -> None:
        """Copy a file from the local machine to the executor.

        Args:
            local_path (str): Path of the file on the local machine.
            remote_path (str): Path of the file on the executor.
        """
        pass

    
class SSHExecutor(Executor):
    def __init__(self, host, user):
        self.host = host
        self.user = user
        self.conn = Connection(host, user=user)
        self._working_dir = '~'
    
    @property
    def working_dir(self):
        return self._working_dir
    
    @working_dir.setter
    def working_dir(self, working_dir):
        self._working_dir = working_dir
    
    def run(self, cmd):
        try:
            with self.conn.cd(self.working_dir):
                result = self.conn.run(cmd, hide=True)
        except Exception as e:
            return '', str(e)
        return result.stdout, result.stderr
    
    def put(self, local_path, remote_path):
        remote_path = f'{self.working_dir}/{remote_path}'
        # SFTP connects to home directory already, so remove the ~/ if exists
        if remote_path.startswith('~/'):
            remote_path = remote_path[2:]
        self.conn.put(local_path, remote_path)

# Note needed for now, but should always be easy to implement
# class LocalExecutor(Executor):
#     def run(self, cmd):
#         pass

    