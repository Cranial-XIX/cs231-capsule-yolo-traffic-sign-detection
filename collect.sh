if [ -z "$1" ]; then
    echo "Usage: ./collect.sh [model name] [file name]"
    echo "Error: No model name"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: ./collect.sh [model name] [file name]"
    echo "Error: No file name"
    exit 1
fi
if [ "$1" != "cnn" ] && [ "$1" != "capsule" ] && [ "$1" != "darknet_d" ] && [ "$1" != "darknet_r" ]  && [ "$1" != "darkcapsule" ]; then
    echo "Invalid model name": "$1"
    exit 1
fi
cp -r runs experiments/$1
zip experiments/$2.zip experiments/$1