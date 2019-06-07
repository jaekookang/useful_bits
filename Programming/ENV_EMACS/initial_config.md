# Initial configuration for the develpment environment

## Install basic programs (for linux)
- Update apt-get list
    + $`sudo apt-get update`
- Install GNU Make
    + $`sudo apt-get install make`
- Install libncurses
    + $`sudo apt-get install libncurses-dev`
- Install tree
    + $`sudo apt-get install tree`

## Install emacs
- Downloa emacs (25.3) from http://gnu.askapache.com/emacs/
- Install emacs
    + $`./configure`
        - For errors (e.g. AppKit), refer to [Link](https://lists.gnu.org/archive/html/bug-gnu-emacs/2016-09/msg00603.html)
        - Do disable gnutls, `./configure --with-gnutls=no`
    + $`make`
        - For `'libxml/tree.h' file not found` error (macOS), see [https://emacs.stackexchange.com/questions/34238/installing-emacs-from-source-make-fatal-error-libxml-tree-h-file-not-found](https://emacs.stackexchange.com/questions/34238/installing-emacs-from-source-make-fatal-error-libxml-tree-h-file-not-found).
    + $`src/emacs -Q`
    + $`make install` (or `sudo make install` if a root permission is required)
    + $`make clean`
    + $`make distclean`
