import subprocess
import multiprocessing
import os
import sys


class LaunchFile:
    def __init__(self, name, package, launch_file):
        self.name = name
        self.package = package
        self.launch_file = launch_file

    def run(self):
        cmd = ['ros2', 'launch', self.package, self.launch_file]
        print(f"Launching [{self.name}] ...")
        subprocess.run(cmd, check=True)

    def launch(self):
        process = multiprocessing.Process(target=self.run)
        process.start()
        return process


class LaunchNode:

    COLOR_INDEX = 0
    COLORS = [
        '\033[31m',  # Red
        '\033[32m',  # Green
        '\033[33m',  # Yellow
        '\033[34m',  # Blue
        '\033[35m',  # Magenta
        '\033[36m',  # Cyan
        '\033[37m',  # White
        '\033[90m',  # Bright Black
        '\033[91m',  # Bright Red
        '\033[92m',  # Bright Green
        '\033[93m',  # Bright Yellow
        '\033[94m',  # Bright Blue
        '\033[95m',  # Bright Magenta
        '\033[96m',  # Bright Cyan
        '\033[97m',  # Bright White
    ]

    RESET_COLOR = '\033[0m'  # Reset color

    def __init__(self, name, package, executable, conda_env, params=None, env_variables=None):
        self.name = name
        self.package = package
        self.executable = executable
        self.conda_env = conda_env
        self.parameters = params if params is not None else {}
        self.env_variables = env_variables if env_variables is not None else {} 
        self.color = self.COLORS[LaunchNode.COLOR_INDEX % len(self.COLORS)]
        LaunchNode.COLOR_INDEX += 1

    def run(self):
        env_defs = [f"{key}={value}" for key, value in self.env_variables.items()]
        param_defs = ["--ros-args"]
        for key, value in self.parameters.items():
            param_defs.append(f"-p")
            param_defs.append(f"{key}:={value}")
        cmd = [
            'conda', 'run', '--no-capture-output', '-n', self.conda_env,
            *env_defs,
            'ros2', 'run',
            self.package,
            self.executable,
            *param_defs
        ]

        print(f"{self.color}{f'Launching [{self.name}] ...'}")
        

        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )

        def print_colored_output(pipe, color, name):
            for line in iter(pipe.readline, ''):
                print(f"{color}[{name}] {line.strip()}{self.RESET_COLOR}")
            pipe.close()

        stdout_thread = multiprocessing.Process(
            target=print_colored_output, 
            args=(process.stdout, self.color, self.name)
        )
        stderr_thread = multiprocessing.Process(
            target=print_colored_output, 
            args=(process.stderr, self.color, self.name)
        )

        stdout_thread.start()
        stderr_thread.start()
        stdout_thread.join()
        stderr_thread.join()
        process.wait()

    def launch(self):
        process = multiprocessing.Process(target=self.run)
        process.start()
        return process


class LaunchConfig:
    def __init__(self, nodes):
        self.nodes = nodes

    def add_node(self, name, package, executable, conda_env, params, env_variables):
        node = LaunchNode(name, package, executable, conda_env, params, env_variables)
        self.nodes.append(node)

    def add_launch_file(self, name, package, launch_file):
        launch_file = LaunchFile(name, package, launch_file)
        self.nodes.append(launch_file)

    def launch_all(self):
        processes = []
        for node in self.nodes:
            process = node.launch()
            processes.append(process)

        for process in processes:
            process.join()
