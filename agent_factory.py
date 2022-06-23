import importlib.util
import importlib
import sys
import os
import common_utils


class AgentFactory:
    @classmethod
    def create(cls, config):
        agent_class = common_utils.relative_import_module_and_get_class(
            config['problem_path'],
            config['agent_module_name'],
            config['agent_class_name'])

        return agent_class(config)
    pass