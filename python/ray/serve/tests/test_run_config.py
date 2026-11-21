import sys
from pathlib import Path
from unittest import mock

import httpx
import pytest
import yaml

import ray
from ray import serve
from ray._common.network_utils import find_free_port
from ray._common.test_utils import wait_for_condition
from ray.serve._private.test_utils import get_application_url
from ray.serve.context import _get_global_client
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import ServeDeploySchema

ECHO_APP = "ray.serve.tests.test_config_files.arg_builders.build_echo_app"
IMPORT_ERROR_APP = "ray.serve.tests.test_config_files.import_error.app"


def make_config(messages: dict) -> dict:
    """Builds a config with one echo application per `{name: message}` entry."""
    return {
        "applications": [
            {
                "name": name,
                "route_prefix": f"/{name}",
                "import_path": ECHO_APP,
                "args": {"message": message},
            }
            for name, message in messages.items()
        ]
    }


def write_config(path: Path, config: dict) -> str:
    path.write_text(yaml.safe_dump(config))
    return str(path)


def app_names() -> set:
    return set(serve.status().applications)


class TestRunConfigInputs:
    @pytest.mark.parametrize(
        "input_type", ["str_path", "path_object", "dict", "schema"]
    )
    def test_accepts_path_dict_and_schema(self, serve_instance, tmp_path, input_type):
        config = make_config({"app1": "one", "app2": "two"})
        if input_type == "str_path":
            arg = write_config(tmp_path / "config.yaml", config)
        elif input_type == "path_object":
            arg = Path(write_config(tmp_path / "config.yaml", config))
        elif input_type == "dict":
            arg = config
        else:
            arg = ServeDeploySchema.model_validate(config)

        handles = serve.run_config(arg)

        # One handle per application, keyed by name and in config order.
        assert list(handles) == ["app1", "app2"]
        assert all(isinstance(h, DeploymentHandle) for h in handles.values())
        assert handles["app1"].remote().result() == "one"
        assert handles["app2"].remote().result() == "two"

    def test_applications_are_reachable_over_http(self, serve_instance):
        serve.run_config(make_config({"app1": "one", "app2": "two"}))

        # The applications are routable as soon as run_config returns.
        assert httpx.get(get_application_url("HTTP", app_name="app1")).text == "one"
        assert httpx.get(get_application_url("HTTP", app_name="app2")).text == "two"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            serve.run_config(str(tmp_path / "does_not_exist.yaml"))

    def test_file_is_not_a_mapping(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- not\n- a mapping\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            serve.run_config(str(path))

    def test_invalid_config(self):
        # An application without an import path fails schema validation.
        with pytest.raises(ValueError):
            serve.run_config({"applications": [{"name": "app1"}]})

    @pytest.mark.parametrize("bad_input", [None, 1, ["applications"]])
    def test_unsupported_type(self, bad_input):
        with pytest.raises(TypeError, match="path to a Serve config file"):
            serve.run_config(bad_input)


class TestRunConfigDeployment:
    def test_deploy_failed_raises(self, serve_instance):
        config = {
            "applications": [{"name": "bad", "import_path": IMPORT_ERROR_APP}],
        }
        with pytest.raises(RuntimeError, match="Deploying application bad failed"):
            serve.run_config(config)

        assert serve.status().applications["bad"].status == "DEPLOY_FAILED"

    def test_returns_after_all_applications_are_running(self, serve_instance):
        handles = serve.run_config(make_config({"app1": "one", "app2": "two"}))

        statuses = serve.status().applications
        assert {name: statuses[name].status for name in handles} == {
            "app1": "RUNNING",
            "app2": "RUNNING",
        }

    def test_redeploying_a_running_app_waits_for_the_update(self, serve_instance):
        """The new config's code and args are live when the call returns."""
        handles = serve.run_config(make_config({"app1": "v1"}))
        assert handles["app1"].remote().result() == "v1"

        handles = serve.run_config(make_config({"app1": "v2"}))
        assert handles["app1"].remote().result() == "v2"

    @pytest.mark.parametrize("blocking", [False, True])
    def test_blocking(self, serve_instance, blocking):
        with mock.patch("ray.serve.api.wait_for_interrupt") as wait_for_interrupt:
            serve.run_config(make_config({"app1": "one"}), blocking=blocking)

        assert wait_for_interrupt.call_count == (1 if blocking else 0)


class TestRunConfigIsDeclarative:
    def test_apps_missing_from_the_new_config_are_deleted(self, serve_instance):
        serve.run_config(make_config({"app1": "one", "app2": "two", "app3": "three"}))
        assert app_names() == {"app1", "app2", "app3"}

        handles = serve.run_config(make_config({"app1": "one", "app3": "three"}))

        assert list(handles) == ["app1", "app3"]
        # Deletion runs in the background, so it can finish after the call returns.
        wait_for_condition(lambda: app_names() == {"app1", "app3"})
        assert handles["app1"].remote().result() == "one"
        assert handles["app3"].remote().result() == "three"

    def test_deleted_app_can_be_deployed_again(self, serve_instance):
        serve.run_config(make_config({"app1": "one", "app2": "two"}))
        serve.run_config(make_config({"app1": "one"}))
        wait_for_condition(lambda: app_names() == {"app1"})

        handles = serve.run_config(make_config({"app1": "one", "app2": "again"}))
        assert handles["app2"].remote().result() == "again"

    def test_apps_started_with_serve_run_are_not_deleted(self, serve_instance):
        @serve.deployment
        def imperative():
            return "imperative"

        serve.run(imperative.bind(), name="imperative", route_prefix="/imperative")
        serve.run_config(make_config({"app1": "one", "app2": "two"}))
        assert app_names() == {"imperative", "app1", "app2"}

        serve.run_config(make_config({"app1": "one"}))

        wait_for_condition(lambda: app_names() == {"imperative", "app1"})
        assert serve.get_app_handle("imperative").remote().result() == "imperative"

    def test_empty_config_deletes_all_declarative_apps(self, serve_instance):
        serve.run_config(make_config({"app1": "one"}))

        assert serve.run_config({"applications": []}) == {}

        wait_for_condition(lambda: app_names() == set())


class TestRunConfigSystemOptions:
    # These tests restart Ray to check what happens when run_config starts Serve, so
    # they invalidate the cluster that `serve_instance` shares. Keep them last.
    def test_options_apply_when_run_config_starts_serve(self, ray_shutdown):
        ray.init(num_cpus=8)
        port = find_free_port()
        config = make_config({"app1": "one"})
        config["proxy_location"] = "HeadOnly"
        config["http_options"] = {"port": port}

        handles = serve.run_config(config)

        assert handles["app1"].remote().result() == "one"
        assert httpx.get(f"http://localhost:{port}/app1").text == "one"
        details = _get_global_client().get_serve_details()
        assert details["http_options"]["port"] == port
        assert details["proxy_location"] == "HeadOnly"

    def test_options_dont_change_a_running_serve(self, ray_shutdown):
        ray.init(num_cpus=8)
        port = find_free_port()
        config = make_config({"app1": "one"})
        config["http_options"] = {"port": port}
        serve.run_config(config)

        # A later config with different system options only updates the apps.
        config = make_config({"app1": "two"})
        config["http_options"] = {"port": find_free_port()}
        handles = serve.run_config(config)

        assert handles["app1"].remote().result() == "two"
        assert _get_global_client().get_serve_details()["http_options"]["port"] == port


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
