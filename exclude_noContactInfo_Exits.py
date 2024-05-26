import sys
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from stem.control import Controller
from stem.util.tor_tools import *
import os

tor_socket_path = os.getenv('TOR_SOCKET_PATH', '/home/-replace-me-/.tor-control.socket')

def get_relays(controller):
    try:
        return controller.get_server_descriptors().run()
    except Exception as e:
        print(f'Failed to get relay descriptors: {e}')
        sys.exit(3)

def process_relays(relays):
    X = []
    y = []
    
    for relay in relays:
        features = [relay.bandwidth_rate, relay.bandwidth_burst, relay.observed_bandwidth, relay.exit_policy.is_exiting_allowed(), int(not bool(relay.contact))]
        
        label = 'exclude' if relay.bandwidth_rate < 1000 else 'keep'
        
        X.append(features)
        y.append(label)
    
    return X, y

def train_model(X_train, y_train):
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    val_score = model.score(X, y)
    cv_scores = cross_val_score(model, X, y, cv=5)
    return val_score, cv_scores

def predict_relays(model, relays):
    exit_excludelist = []
    
    for idx, features in enumerate(X):
        prediction = model.predict([features])[0]
        
        if prediction == 'exclude':
            relay = relays[idx]
            if is_valid_fingerprint(relay.fingerprint):
                exit_excludelist.append(relay.fingerprint)
                print("Excluding relay with fingerprint: %s" % relay.fingerprint)
            else:
                print('Invalid Fingerprint: %s' % relay.fingerprint)
    
    return exit_excludelist

def main():
    try:
        with Controller.from_socket_file(path=tor_socket_path) as controller:
            controller.authenticate()
            
            if not controller.is_set('UseMicrodescriptors'):
                print('"UseMicrodescriptors 0" is required in your torrc configuration. Exiting.')
                sys.exit(2)
            
            if controller.is_set('ExcludeExitNodes'):
                print('ExcludeExitNodes is in use already. Exiting.')
                sys.exit(4)
            
            relays = get_relays(controller)
            
            X, y = process_relays(relays)
            
            X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
            
            model = train_model(X_train, y_train)
            
            val_score, cv_scores = evaluate_model(model, X_val, y_val)
            print(f"Validation score: {val_score:.2f}")
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean cross-validation score: {cv_scores.mean():.2f}")
            
            test_score = evaluate_model(model, X_test, y_test)[0]
            print(f"Test set score: {test_score:.2f}")
            
            exit_excludelist = predict_relays(model, relays)
            
            controller.set_conf('ExcludeExitNodes', exit_excludelist)
            print('##################################################################################')
            print('Excluded a total of %s exit relays based on the model predictions.' % len(exit_excludelist))
            print('This tor configuration change is not permanently stored (non-persistent). A Tor Browser restart will revert this change.')
    except Exception as e:
        print(f'An error occurred: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
