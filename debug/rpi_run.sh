sudo apt -y update
sudo apt install -y build-essential git libprotobuf-dev python3-pip libprotobuf-c-dev protobuf-c-compiler protobuf-compiler libcap-dev libnl-3-dev libnl-genl-3-dev libnet1-dev libbsd-dev libnl-route-3-dev libnfnetlink-dev pkg-config asciidoctor

git clone  https://github.com/checkpoint-restore/criu.git criu
cd criu
make clean
make
make install

sudo apt install -y linux-tools-5.11.0-1007-raspi linux-tools-raspi