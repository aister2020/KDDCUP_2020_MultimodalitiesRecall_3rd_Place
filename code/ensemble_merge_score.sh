
models=("20200605102104" "20200605102155" "20200605102210" "20200605103300" "20200605103317" "20200605103338" "20200605103358" "20200605104920" "20200605104935" "20200605104949" "20200605105004" "20200605110558" "20200605110612" "20200605110629")

allmodel=""
for model in ${models[@]}
do    
    for dataset in {1..1}
    do
        
        if [ -f "../user_data/output/${model}_${dataset}_cv0" ];then
            echo ${model}_${dataset}
            if [ -n "$allmodel" ]; then
                allmodel=${allmodel},
            fi
        
            rm ../user_data/output/${model}_${dataset}_cvmerged
            python cv_merge_score.py ../user_data/output/${model}_${dataset}_cv 5
            allmodel=${allmodel}../user_data/output/${model}_${dataset}_cvmerged
        fi
        
    done
done

echo  $allmodel
python file_merge_score.py $allmodel ../user_data/output/interid_nn_merged



models=("0605115205" "0605115428" "0605115503" "0605115521" "0605115547" "0605122723" "0605122757" "0605122825" "0605122841" "0605122902" "0605130536" "0605130558" "0605130617" "0605130634")

allmodel=""
for model in ${models[@]}
do    
        
    if [ -f "../user_data/output/testA_lgb/lgb${model}" ];then
        echo ${model}
        if [ -n "$allmodel" ]; then
            allmodel=${allmodel},
        fi
    
        allmodel=${allmodel}../user_data/output/testA_lgb/lgb${model}
    fi
        
done

echo  $allmodel
python file_merge_score.py $allmodel ../user_data/output/interid_lgb_merged



models=("20200605135828_20200528202620" "20200605135844_20200527214553" "20200605135855_20200528023614" "20200605135905_20200527235736" "20200605141133_20200529114748" "20200605141145_20200528154733" "20200605141200_20200528125945" "20200605141213_20200529120120" "20200605142455_20200529120105" "20200605142514_20200529115623" "20200605142526_20200530144344" "20200605142545_20200529014325" "20200605150057_20200530085657" "20200605150632_20200530085707")

allmodel=""
for model in ${models[@]}
do    
    for dataset in {1..1}
    do
        
        if [ -f "../user_data/output/${model}_${dataset}_cv0" ];then
            echo ${model}_${dataset}
            if [ -n "$allmodel" ]; then
                allmodel=${allmodel},
            fi
        
            rm ../user_data/output/${model}_${dataset}_cvmerged
            python cv_merge_score.py ../user_data/output/${model}_${dataset}_cv 5
            allmodel=${allmodel}../user_data/output/${model}_${dataset}_cvmerged
        fi
        
    done
done

echo  $allmodel
python file_merge_score.py $allmodel  ../user_data/output/kn_nn_merged


models=("0605151550" "0605151621" "0605151638" "0605151654" "0605154950" "0605155010" "0605155026" "0605155044" "0605165244" "0605165302" "0605165318" "0605165336" "0605175627" "0605175650")

allmodel=""
for model in ${models[@]}
do    
        
    if [ -f "../user_data/output/testA_lgb/lgb${model}" ];then
        echo ${model}
        if [ -n "$allmodel" ]; then
            allmodel=${allmodel},
        fi
    
        allmodel=${allmodel}../user_data/output/testA_lgb/lgb${model}
    fi
        
done

echo  $allmodel
python file_merge_score.py $allmodel ../user_data/output/kn_lgb_merged




python file_merge_score_wei.py ../user_data/output/interid_nn_merged,../user_data/output/interid_lgb_merged 5.3,4.7 ../user_data/output/merge1
python file_merge_score_wei.py ../user_data/output/kn_nn_merged,../user_data/output/kn_lgb_merged 5.3,4.7 ../user_data/output/merge2
python file_merge_score_wei.py ../user_data/output/merge1,../user_data/output/merge2 8,2 ../user_data/output/merge3

python gen_submit.py ../user_data/output/merge3 ../prediction_result/submission.csv

