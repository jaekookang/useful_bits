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
    + $`src/emacs -Q`
    + $`make install` or `sudo make install` if a root permission is required
    + $`make clean`
    + $`make distclean`
