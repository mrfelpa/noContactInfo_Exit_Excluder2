
This is an enhanced version of the project developed by [nusenu](https://github.com/nusenu/noContactInfo_Exit_Excluder)


# Requirements

- Python 3.6 or higher
- scikit-Learn
- Stem
- Browser Tor
- Tor settings with the UseMicrodescriptors 0 option enabled in the torrc file.

# Installing

Clone the repository:

        git clone https://github.com/mrfelpa/ContactInfo_Exit_Excluder2.git
        
        cd ContactInfo_Exit_Excluder2

Instale as dependências:

        pip install -r requirements.txt

# Configuration

Make sure that Tor Browser is running and configured to allow connections through the specified socket. In the torrc file, add or edit the following line:

        UseMicrodescriptors 0

Update the socket path in script to the correct path of your environment:

        
        torsocketpath = '/home/-replace-me-/.tor-control.socket'

Run the main script:

        exclude_noContactInfo_Exits.py

# Model

- The extracted data is divided into training, validation and test sets. ***A Stochastic Descending Gradient (SGD) classifier*** is trained to predict which relays to exclude.

- The model is evaluated using cross-validation and in the test set. Performance metrics are printed to the console.

- The model is used to predict which relays should be deleted and the configuration is temporarily applied in Tor.

# Contributing

We value contributions, for improvements and fixes, open an Issue or send a pull request for improvements and fixes.
