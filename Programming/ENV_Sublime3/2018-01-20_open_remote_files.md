# Open remote files on server using Sublime3

* 2018-01-20
* ref:
	- [Link](https://stackoverflow.com/questions/37458814/how-to-open-remote-files-in-sublime-text-3)
* Note
	- tested for macOS and Linux

## (1) Download rsub on server

```bash
SRC=/home/jk/programs
wget -O $SRC https://raw.github.com/aurora/rmate/master/rmate
chmod a+x $SRC
export PATH="$SRC:$PATH"
```

## (2) Install Sublime3 package on local

- Install rsub using Package Manager (`CMD + Shift + P` on Mac)

## (3) Connect to server using port=52698

```bash
ssh -R 52698:localhost:52698 server_user@server_address
```

## (4) Open file on server

```bash
rsub path_to_file/file
```