import sys
import os
import logging
import numpy as np
import secrets
import stat
import subprocess
import warnings
import argparse
import time
import asyncio
from typing import List, Tuple
import keyring
import yaml

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from stem.control import Controller
from stem.util.tor_tools import is_valid_fingerprint
from stem.descriptor.server_descriptor import ServerDescriptor

TOR_SOCKET_PATH_KEY = "tor_socket_path"
TOR_CONTROL_PASSWORD_KEY = "tor_control_password"
DEFAULT_CONFIG_PATH = "config.yaml"

BANDWIDTH_THRESHOLD = 1000
RANDOM_STATE = secrets.randbelow(1000)
CROSS_VALIDATION_SPLITS = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.5

FEATURE_BANDWIDTH_RATE = 0
FEATURE_BANDWIDTH_BURST = 1
FEATURE_OBSERVED_BANDWIDTH = 2
FEATURE_EXIT_POLICY = 3
FEATURE_CONTACT = 4

MAX_RETRIES = 5
INITIAL_BACKOFF = 1


class RedactingFormatter(logging.Formatter):
    def filter_sensitive_data(self, record: logging.LogRecord) -> logging.LogRecord:
        message = record.getMessage()
        message = message.replace(keyring.get_password("tor_config", TOR_SOCKET_PATH_KEY) or "", "<TOR_SOCKET_PATH_REDACTED>")
        message = message.replace("fingerprint: ", "fingerprint: <REDACTED>")
        record.msg = message
        return record

    def format(self, record: logging.LogRecord) -> str:
        record = self.filter_sensitive_data(record)
        return super().format(record)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
formatter = RedactingFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)


def check_socket_permissions(socket_path: str) -> bool:
    try:
        file_stat = os.stat(socket_path)
        if file_stat.st_uid == os.getuid():
            logging.info(f"User is the owner of Tor control socket: {socket_path}")
            return True

        import grp
        import pwd

        try:
            socket_group_name = grp.getgrgid(file_stat.st_gid).gr_name
            user_name = pwd.getpwuid(os.getuid()).pw_name
            result = subprocess.run(['id', '-Gn', user_name], capture_output=True, text=True, check=True)
            user_groups = result.stdout.split()

            if socket_group_name in user_groups:
                logging.info(f"User is a member of group {socket_group_name} which owns the Tor control socket.")
                return True

        except KeyError:
            logging.warning("Socket's Group doesn't exist")
            return False
        except Exception as e:
            logging.warning(f"Failed to check group membership: {e}")
            return False

        logging.warning(f"User does not own, nor is in the group that owns Tor control socket: {socket_path}")
        return False

    except FileNotFoundError:
        logging.error(f"Tor control socket not found: {socket_path}")
        raise
    except OSError as e:
        logging.error(f"Error checking Tor control socket permissions: {e}")
        raise


async def connect_to_tor(socket_path: str, password: str = None) -> Controller:
    for attempt in range(MAX_RETRIES):
        try:
            controller = Controller.from_socket_file(path=socket_path)
            if password:
                controller.authenticate(password=password)
            else:
                controller.authenticate()
            return controller
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed to connect to Tor controller: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(INITIAL_BACKOFF * (2**attempt))
            else:
                raise


async def get_relays(controller: Controller) -> List[ServerDescriptor]:
    for attempt in range(MAX_RETRIES):
        try:
            return controller.get_server_descriptors().run()
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed to get relay descriptors: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(INITIAL_BACKOFF * (2**attempt))
            else:
                raise


def process_relays(relays: List[ServerDescriptor]) -> Tuple[List[List[float]], List[str]]:
    X = []
    y = []

    for relay in relays:
        try:
            features = [
                float(relay.bandwidth_rate),
                float(relay.bandwidth_burst),
                float(relay.observed_bandwidth),
                int(relay.exit_policy.is_exiting_allowed()),
                int(not bool(relay.contact)),
            ]

            label = 'exclude' if relay.bandwidth_rate < BANDWIDTH_THRESHOLD else 'keep'

            X.append(features)
            y.append(label)
        except (AttributeError, ValueError, TypeError) as e:
            logging.warning(f"Skipping relay (fingerprint redacted) due to processing error: {e}")

    return X, y


def create_model() -> Pipeline:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=RANDOM_STATE))
    ])
    return pipeline


def train_model(model: Pipeline, X_train: List[List[float]], y_train: List[str]) -> Pipeline:
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Failed to train the model: {e}")
        raise


def evaluate_model(model: Pipeline, X: List[List[float]], y: List[str]) -> Tuple[float, np.ndarray]:
    try:
        val_score = model.score(X, y)
        cv_scores = cross_val_score(model, X, y, cv=CROSS_VALIDATION_SPLITS)
        return val_score, cv_scores
    except Exception as e:
        logging.error(f"Failed to evaluate the model: {e}")
        raise


def predict_relays(model: Pipeline, relays: List[ServerDescriptor], X: List[List[float]]) -> List[str]:
    exit_excludelist = []

    for idx, features in enumerate(X):
        try:
            prediction = model.predict([features])[0]

            if prediction == 'exclude':
                relay = relays[idx]
                if is_valid_fingerprint(relay.fingerprint):
                    exit_excludelist.append(relay.fingerprint)
                    logging.info("Excluding relay with fingerprint: <REDACTED>")
                else:
                    logging.warning('Invalid Fingerprint: <REDACTED>')
        except Exception as e:
            logging.warning(f"Failed to predict relay (fingerprint redacted) due to error: {e}")

    return exit_excludelist


async def configure_tor_exits(controller: Controller, exit_excludelist: List[str]) -> None:
    for attempt in range(MAX_RETRIES):
        try:
            controller.set_conf('ExcludeExitNodes', ','.join(exit_excludelist))
            logging.info('Excluded a total of %s exit relays based on the model predictions.', len(exit_excludelist))
            logging.info('This Tor configuration change is not permanently stored (non-persistent). A Tor Browser restart will revert this change.')
            return
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed to configure Tor exits: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(INITIAL_BACKOFF * (2**attempt))
            else:
                raise


async def check_tor_configuration(controller: Controller) -> None:
    try:
        use_microdescriptors = controller.get_conf("UseMicrodescriptors")
        if use_microdescriptors != ['0']:
            raise ValueError('"UseMicrodescriptors 0" is required in your torrc configuration.')

        exclude_exit_nodes = controller.get_conf("ExcludeExitNodes")
        if exclude_exit_nodes:
            logging.warning(f"ExcludeExitNodes is already set to {exclude_exit_nodes}. This script will overwrite it.")

    except Exception as e:
        logging.error(f"Failed to check Tor configuration: {e}")
        raise


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}, using defaults and keyring.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise


async def main() -> None:
    parser = argparse.ArgumentParser(description='Tor Exit Node Configuration Script')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        socket_path = keyring.get_password("tor_config", TOR_SOCKET_PATH_KEY) or os.getenv("TOR_SOCKET_PATH")
        control_password = keyring.get_password("tor_config", TOR_CONTROL_PASSWORD_KEY)
        if not socket_path:
             raise ValueError("Tor Socket path not found in keyring or environment variables")

        if not check_socket_permissions(socket_path):
            logging.error(f"Incorrect permissions on Tor control socket: {socket_path}. Please ensure the script is run by the owner or a member of the appropriate group.")
            sys.exit(5)

        async with await connect_to_tor(socket_path, control_password) as controller:
            try:
                await check_tor_configuration(controller)
            except ValueError as e:
                logging.error(e)
                sys.exit(2)

            relays = await get_relays(controller)
            X, y = process_relays(relays)

            X_train, X_val_test, y_train, y_val_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_val_test, y_val_test, test_size=VAL_SIZE / TEST_SIZE, random_state=RANDOM_STATE, stratify=y_val_test
            )

            model = create_model()

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = train_model(model, X_train, y_train)

            val_score, cv_scores = evaluate_model(model, X_val, y_val)
            logging.info("Validation score: %.2f", val_score)
            logging.info("Cross-validation scores: %s", cv_scores)
            logging.info("Mean cross-validation score: %.2f", cv_scores.mean())

            test_score = evaluate_model(model, X_test, y_test)[0]
            logging.info("Test set score: %.2f", test_score)

            exit_excludelist = predict_relays(model, relays, X)

            await configure_tor_exits(controller, exit_excludelist)

    except Exception as e:
        logging.error('An error occurred: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
