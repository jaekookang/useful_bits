************************************************
Install brew and GNU command line tools on macOS
************************************************

- credit to:
	- [Top Bug Net](https://www.topbug.net/blog/2013/04/14/install-and-use-gnu-command-line-tools-in-mac-os-x/)


Install Homebrew
----------------
.. code-block:: python
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Add path to .bashrc
-------------------
.. code-block:: python
	export PATH="$(brew --prefix coreutils)/libexec/gnubin:/usr/local/bin:$PATH"

Install GNU command line tools
------------------------------

* First of all,
.. code-block:: python
	brew install coreutils

Install additional tools
------------------------
.. code-block:: python
	brew install binutils
	brew install diffutils
	brew install ed --with-default-names
	brew install findutils --with-default-names
	brew install gawk
	brew install gnu-indent --with-default-names
	brew install gnu-sed --with-default-names
	brew install gnu-tar --with-default-names
	brew install gnu-which --with-default-names
	brew install gnutls
	brew install grep --with-default-names
	brew install gzip
	brew install screen
	brew install watch
	brew install wdiff --with-gettext
	brew install wget

