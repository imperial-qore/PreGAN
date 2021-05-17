sudo apt update
sudo apt install build-essential
sudo apt install libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler python-protobuf
sudo apt-get install libcap-dev
sudo apt-get install libnl-3-dev libnl-genl-3-dev
sudo apt install libnet1-dev libbsd-dev
sudo apt-get install asciidoctor

git clone  https://github.com/xemul/criu.git criu
cd criu
make clean
make
make install