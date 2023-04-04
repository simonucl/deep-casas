for ((i=1; i<=30; i++))
do
    # Unix command here #
    i1=$i
    if (($i < 10)); then
        i1="0${i}";
    fi
    echo $i1
    wget http://casas.wsu.edu/datasets/hh1$i1.zip
    unzip hh1$i1.zip
done