import sys
import os
import logging
import numpy as np
import secrets  # For generating secure random numbers
import stat  # For checking file permissions
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings
from stem.control import Controller
from stem.util.tor_tools import is_valid_fingerprint


class RedactingFormatter(logging.Formatter):
  
    def filter_sensitive_data(self, record):
        message = record.getMessage()
        # Simple redaction - replace fingerprints (adjust regex as needed)
        message = message.replace(TOR_SOCKET_PATH, "<TOR_SOCKET_PATH_REDACTED>") 
        message = message.replace("fingerprint: ", "fingerprint: <REDACTED>") 
        record.msg = message
        return record

    def format(self, record):
      record = self.filter_sensitive_data(record)
      return super().format(record)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
formatter = RedactingFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)


TOR_SOCKET_PATH = os.getenv('TOR_SOCKET_PATH', '/home/-replace-me-/.tor-control.socket')
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

def check_socket_permissions(socket_path):
    
    try:
        file_stat = os.stat(socket_path)
        # Check if the current user is the owner
        if file_stat.st_uid == os.getuid():
            logging.info(f"User is the owner of Tor control socket: {socket_path}")
            return True
        # Check if the current user is in the group that owns the socket
        import grp
        try:
            socket_group_name = grp.getgrgid(file_stat.st_gid).gr_name
            import pwd
            user_name = pwd.getpwuid(os.getuid()).pw_name
            import subprocess
            result = subprocess.run(['groups', user_name], capture_output=True, text=True)
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

def connect_to_tor(socket_path):
  
    try:
        controller = Controller.from_socket_file(path=socket_path)
        controller.authenticate()
        return controller
    except Exception as e:
        logging.error(f"Failed to connect to Tor controller: {e}")
        raise

def get_relays(controller):
    
    try:
        return controller.get_server_descriptors().run()
    except Exception as e:
        logging.error(f"Failed to get relay descriptors: {e}")
        raise

def process_relays(relays):

    X = []
    y = []

    for relay in relays:
        try:
            features = [
                relay.bandwidth_rate,
                relay.bandwidth_burst,
                relay.observed_bandwidth,
                relay.exit_policy.is_exiting_allowed(),
                int(not bool(relay.contact))
            ]

            label = 'exclude' if relay.bandwidth_rate < BANDWIDTH_THRESHOLD else 'keep'

            X.append(features)
            y.append(label)
        except Exception as e:
            logging.warning(f"Skipping relay (fingerprint redacted) due to processing error: {e}") 

    return X, y


def create_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=RANDOM_STATE))
    ])
    return pipeline


def train_model(model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Failed to train the model: {e}")
        raise

def evaluate_model(model, X, y):
    try:
        val_score = model.score(X, y)
        cv_scores = cross_val_score(model, X, y, cv=CROSS_VALIDATION_SPLITS)
        return val_score, cv_scores
    except Exception as e:
        logging.error(f"Failed to evaluate the model: {e}")
        raise

def predict_relays(model, relays, X):
    
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

def configure_tor_exits(controller, exit_excludelist):
  
    try:
        controller.set_conf('ExcludeExitNodes', exit_excludelist)
        logging.info('Excluded a total of %s exit relays based on the model predictions.', len(exit_excludelist))
        logging.info('This Tor configuration change is not permanently stored (non-persistent). A Tor Browser restart will revert this change.')
    except Exception as e:
        logging.error(f"Failed to configure Tor exits: {e}")
        raise


def check_tor_configuration(controller):
    
    if not controller.is_set('UseMicrodescriptors'):
        raise ValueError('"UseMicrodescriptors 0" is required in your torrc configuration.')

    if controller.is_set('ExcludeExitNodes'):
        raise ValueError('ExcludeExitNodes is in use already. Please remove/comment it in your torrc.')

def main():
    
    try:
        # Check Tor control socket permissions
        if not check_socket_permissions(TOR_SOCKET_PATH):
            logging.error(f"Incorrect permissions on Tor control socket: {TOR_SOCKET_PATH}. Please ensure the script is run by the owner or a member of the appropriate group.")
            sys.exit(5)
        with connect_to_tor(TOR_SOCKET_PATH) as controller:
            try:
                check_tor_configuration(controller)
            except ValueError as e:
                logging.error(e)
                sys.exit(2)

            relays = get_relays(controller)
            X, y = process_relays(relays)

            # Split data into training, validation, and test sets
            X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y) #Stratify y to keep proportions
            X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_val_test) #Stratify y_val_test to keep proportions

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

            configure_tor_exits(controller, exit_excludelist)

    except Exception as e:
        logging.error('An error occurred: %s', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
