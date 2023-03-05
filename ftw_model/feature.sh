for ((i=1; i<=30; i++))
do
    # Unix command here #
    i1=$i
    if (($i < 10)); then
        i1="0${i}";
    fi
    if (($i != 2)); then
        continue
    fi
    echo $i1
    # python extract_feature.py --input "../hh_dataset/hh1$i1/hh1$i1.ann.txt" --delta 20 --window ESTWs --output "../hh_dataset/hh_npy/estw_hh1$i1.npy"
    # python extract_feature.py --input "../hh_dataset/hh1$i1/hh1$i1.ann.txt" --delta 20 --window SESTWs --output "../hh_dataset/hh_npy/sestw_hh1$i1.npy"
    python extract_feature.py --input "../hh_dataset/hh1$i1/hh1$i1.ann.txt" --delta 20 --window FIB_FTWs --output "../hh_dataset/hh_npy/fib_hh1$i1.npy"

    # python train.py --model BiLSTM --features ../hh_dataset/hh_npy/estw_hh1${i1}_feature.npy --activities ../hh_dataset/hh_npy/estw_hh1${i1}_activity.npy --feature_encoding mean_with_weight
    # python train.py --model BiLSTM --features ../hh_dataset/hh_npy/sestw_hh1${i1}_feature.npy --activities ../hh_dataset/hh_npy/sestw_hh1${i1}_activity.npy --feature_encoding mean_with_weight
    # python train.py --model BiLSTM --features ../hh_dataset/hh_npy/fib_hh1${i1}_feature.npy --activities ../hh_dataset/hh_npy/fib_hh1${i1}_activity.npy --feature_encoding mean_with_weight

done
