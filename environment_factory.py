import common_utils


class EnvironmentFactory:
    
    @classmethod
    def create(cls, config):
        env_class = common_utils.relative_import_module_and_get_class(
            config['problem_path'],
            config['env_module_name'],
            config['env_class_name'])

        return env_class(config)