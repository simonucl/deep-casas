encoding_methods=('mean_with_weight' '1d_cnn')
encoding_methods=('1d_cnn')
batch_size=(32)
# time encoding from 1 - 8
time_encoding=('1' '2' '3' '4' '5' '6' '7' '8')
deltas=(10)
# encoding_methods=('1d_cnn')
for ((i=1; i<=12; i++))
do            
    # Unix command here #
    i1=$i
    if (($i < 10)); then
        i1="0${i}";
    fi
    # # run only when i is not 01, 02, 03, 09, 13, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30
    # if (($i == 1 || $i == 2 || $i == 3 || $i == 9 || $i == 13 || $i == 15 || $i == 17 || $i == 18 || $i == 19 || $i == 21 || $i == 23 || $i == 24 || $i == 27 || $i == 28 || $i == 29 || $i == 30)); then
    #     continue
    # fi
    
    # run only when i equals to 2, 9 and 12
    if (($i != 2 && $i != 9 && $i != 12)); then
        continue
    fi

    echo $i1
# python extract_feature.py --input "../hh_dataset/hh1$i1/hh1$i1.ann.txt" --delta 20 --window ESTWs --output "../hh_dataset/hh_npy/estw_hh1$i1.npy"
# python extract_feature.py --input "../hh_dataset/hh1$i1/hh1$i1.ann.txt" --delta 20 --window SESTWs --output "../hh_dataset/hh_npy/sestw_hh1$i1.npy"
    for delta in ${deltas[@]}; do
        # python extract_feature.py --input "../hh_dataset/hh1$i1/hh1$i1.ann.txt" --delta ${delta} --window FIB_FTWs --output "../hh_dataset/hh_npy/fib_hh1$i1.npy"
        for feature_encoding in ${encoding_methods[@]}; do
            for batch in ${batch_size[@]}; do
                for time_encoding in ${time_encoding[@]}; do
                    python train.py --model BiLSTM --features ../hh_dataset/hh_npy/fib_hh1${i1}_feature.npy --activities ../hh_dataset/hh_npy/fib_hh1${i1}_activity.npy --feature_encoding ${feature_encoding} --delta ${delta} --file_ext "${time_encoding}_batch_first" --batch ${batch} --time_encoding ${time_encoding}
                done
            done
            # python train.py --model BiLSTM --features ../hh_dataset/hh_npy/fib_hh1${i1}_feature.npy --activities ../hh_dataset/hh_npy/fib_hh1${i1}_activity.npy --feature_encoding ${feature_encoding} --delta ${delta} --file_ext '_unmerged_with_time2vec'
        done
    done
done
