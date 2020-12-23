from ruamel import yaml

from constants import (
    YAML_VERSION,
    DEFAULT_ENCODING
)

def _is_ascii(text):
    return all(ord(char) for char in text)

def fix_yaml_loader():
    """Ensure that any string read by yaml is represented as unicode."""

    def construct_yaml_str(self, node):
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)

    yaml.Loader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)

def replace_enviroment_variables():
    def replace_environment_variables() -> None:
        """Enable yaml loader to process the environment variables in the yaml."""
        import re
        import os

        # eg. ${USER_NAME}, ${PASSWORD}
        env_var_pattern = re.compile(r"^(.*)\$\{(.*)\}(.*)$")
        yaml.add_implicit_resolver("!env_var", env_var_pattern)

        def env_var_constructor(loader, node):
            """Process environment variables found in the YAML."""
            value = loader.construct_scalar(node)
            expanded_vars = os.path.expandvars(value)
            if "$" in expanded_vars:
                not_expanded = [w for w in expanded_vars.split() if "$" in w]
                raise ValueError(
                    "Error when trying to expand the environment variables "
                    "in '{}'. Please make sure to also set these environment "
                    "variables: '{}'.".format(value, not_expanded)
                )
            return expanded_vars

        yaml.SafeConstructor.add_constructor("!env_var", env_var_constructor)

def read_yaml(content):
    fix_yaml_loader()

    replace_enviroment_variables()

    yaml_parser = yaml.YAML(typ='safe')
    yaml_parser.version = YAML_VERSION

    if _is_ascii(content):
        content = (
            content.encode('utf-8')
            .decode('raw_unicode_escape')
            .encode('utf-16', 'surrogatepass')
            .decode('utf-16')
        )

    return yaml_parser.load(content) or {}

def read_file(filename, encoding=DEFAULT_ENCODING):
    try:
        with open(filename, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f'File "{filename}" does not exist.')

def read_config_file(filename):
    content = read_yaml(read_file(filename))

    if content is None:
        return {}
    elif isinstance(content, dict):
        return content
    else:
        raise ValueError(
            'Tried to load invalid config file "{}". '
            'Expected a key value mapping but found {}'
            .format(filename, content)
        )
