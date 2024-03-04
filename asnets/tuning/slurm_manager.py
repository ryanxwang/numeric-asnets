"""Class to handle submission, waiting, and polling of jobs to slurm"""
from dataclasses import dataclass
from enum import Enum
import os
import subprocess
import time
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from executor import Executor

# NOTE: define environment and job config separately such that the environment
# can be reused for multiple jobs

@dataclass
class Environment:
    partition: str
    qos: str
    output_file: str = 'S_%x_%j.out'
    error_file: str = 'S_%x_%j.err'


@dataclass
class JobConfig:
    name: str
    time: Tuple[int, int, int]  # (hours, minutes, seconds)
    mem: int  # in GB
    ntasks: int = 1
    cpus_per_task: int = 1


class JobStatus(Enum):
    INACTIVE = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3


@dataclass
class Job:
    cmds: List[str]
    config: JobConfig
    env: Environment
    job_id: Optional[str] = None
    status: JobStatus = JobStatus.INACTIVE
    job_file: Optional[str] = None
        

class SlurmManager:
    def __init__(self, executor: Executor, job_dir: str = 'jobs'):
        self.executor = executor
        self.job_dir: str = job_dir
        self.email: Optional[str] = None
        self.email_conds: Set[str] = set()
        self.jobs: Dict[str, Job] = {}

        self.executor.run(f'mkdir -p {job_dir}')


    def set_email(self, email: str, conds: Set[str] = {'END', 'FAIL'}) -> None:
        """Set email address to send notifications to.
        
        Args:
            email (str): Email address to send notifications to
            conds (Set[str], optional): Set of conditions to send
            notifications for. Defaults to {'END', 'FAIL'}.
        """
        self.email = email
        self.email_conds = email_conds


    def submit(self, job: Job) -> str:
        """Submit a job to slurm. The job_id fiend of the job will be set to the
        job ID returned by slurm, and the job_file field will be set to the path
        of the job script.

        Args:
            job (Job): Job to submit
        
        Returns:
            str: Job ID
        """
        assert job.status == JobStatus.INACTIVE and job.job_id is None \
            and job.job_file is None, 'Job already submitted'
        
        # generate slurm script
        script = self._generate_script(job)

        # write the script to the job directory, using a job name of format
        # S_<job_name>_<uuid>.sh. The ideal thing to use is the job ID which we
        # don't know yet. Will rename after submission. The S_ prefix helps
        # with gitignore
        job_file = os.path.join(
            self.job_dir, f'{job.config.name}_{uuid4()}.slurm')
        self.executor.run(f'echo "{script}" > {job_file}')
        # set the job file field even though we will rename later, this helps
        # debugging if something goes wrong
        job.job_file = job_file

        # submit the job, don't do error handling
        job_id = self.executor.run(f'sbatch {job_file}')[0]
        job_id = job_id.split()[-1].strip()
        job.job_id = job_id
        job.status = JobStatus.PENDING

        # rename the job file to include the job ID
        job_file_new = os.path.join(self.job_dir,
                                    f'S_{job_id}-{job.config.name}.slurm')
        self.executor.run(f'mv {job_file} {job_file_new}')
        job.job_file = job_file_new
        
        # add job to active jobs
        self.jobs[job_id] = job
        return job_id
    

    def poll(self, job_id: str) -> JobStatus:
        """Poll the status of a job.

        Args:
            job_id (str): Job ID to poll

        Returns:
            JobStatus: Status of the job
        """
        assert job_id in self.jobs, 'Job not found'
        job = self.jobs[job_id]
        assert job.status != JobStatus.COMPLETED, 'Job already completed'
        
        # get job status
        job_status = self.executor.run(f'squeue -h -j {job_id}')[0]

        # parse job status
        job_status = job_status.strip().split()
        if len(job_status) < 5:
            # job not found, assume completed
            job.status = JobStatus.COMPLETED
            return job.status
        job_status = job_status[4]
        if job_status == 'PD':
            job.status = JobStatus.PENDING
        elif job_status == 'R':
            job.status = JobStatus.RUNNING
        elif job_status in ['CD', 'CG', 'F', 'OOM', 'TO']:
            job.status = JobStatus.COMPLETED
        else:
            # something went wrong, just mark as completed
            job.status = JobStatus.COMPLETED
            print(f'Unknown job status {job_status}')
        
        return job.status
    

    def wait(self, job_id: str, poll_interval: float = 10) -> None:
        """Wait for a job to complete. The job status will be polled every
        poll_interval seconds.

        Args:
            job_id (str): Job ID to wait for
            poll_interval (float): Poll interval in seconds. Defaults to 10.
        """
        while True:
            job_status = self.poll(job_id)
            if job_status == JobStatus.COMPLETED:
                break
            time.sleep(poll_interval)
    
    
    def cancel(self, job_id: str) -> None:
        """Cancel a job.

        Args:
            job_id (str): Job ID to cancel
        """
        assert job_id in self.jobs, 'Job not found'
        job = self.jobs[job_id]
        assert job.status != JobStatus.COMPLETED, 'Job already completed'
        self.executor.run(f'scancel {job_id}')
        job.status = JobStatus.COMPLETED
        
    
    def _generate_script(self, job: Job) -> str:
        """Generate slurm script for a job.

        Args:
            job (Job): Job to generate script for

        Returns:
            str: Slurm script
        """
        script = ['#!/bin/bash']
        
        # handle the job config
        script.append(f'#SBATCH --job-name={job.config.name}')
        script.append('#SBATCH --time={:02d}:{:02d}:{:02d}'.format(
            *job.config.time))
        script.append(f'#SBATCH --mem={job.config.mem}G')
        script.append(f'#SBATCH --ntasks={job.config.ntasks}')
        script.append(f'#SBATCH --cpus-per-task={job.config.cpus_per_task}')

        # handle the environment
        script.append(f'#SBATCH --partition={job.env.partition}')
        script.append(f'#SBATCH --qos={job.env.qos}')
        script.append(f'#SBATCH --output={job.env.output_file}')
        script.append(f'#SBATCH --error={job.env.error_file}')

        # handle email notifications
        if self.email is not None:
            assert len(self.email_conds) > 0, \
                'Must specify at least one email condition'
            script.append(f'#SBATCH --mail-user={self.email}')
            script.append(f'#SBATCH --mail-type={",".join(self.email_conds)}')
        
        script.append('\n')
        script.extend(job.cmds)
        
        return '\n'.join(script)
        