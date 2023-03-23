for ((i=0; i<=30; i++))
do
    # Unix command here #
    i1=$i
    if (($i < 10)); then
        i1="0${i}";
    fi
    # skip if not equal to 06, 13, 15, 17
    if [ $i1 != "06" ] && [ $i1 != "13" ] && [ $i1 != "15" ] && [ $i1 != "17" ]; then
        continue
    fi
    i1="1$i1"
    echo $i1
    python combine_act.py --hh_dataset ${i1}
done