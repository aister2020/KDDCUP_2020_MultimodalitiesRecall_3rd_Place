#!/bin/bash

####################################################################################################################
#
# 说明：
#   无论单机版本还是分布式版本，请使用本脚本启动您的程序。本脚本默认项目具有一个run.py的主入口。
#   脚本通过用户输入的参数确认执行本地版还是AFO版本。脚本默认从conf文件夹内加载配置。
#       * 对于local（默认）模式，从local_settings中以及hyper_params文件夹中抽取程序运行参数，抽取后输送给run.py执行
#       * 对于afo模式，加载afo_settings.xml文件，并将指定的model配置文件路径添加入afo_settings.xml，传递给
#         afo进行执行
#
# 用法：
#   sh run.sh [ --mode local | afo ] [ --model ] [ --tag ]
#
# 参数说明：
#   mode    必传参数。有local（本地版）和afo（分布式）两种模式。
#   model   必传参数。模型的配置全部写在这个文件内，模型配置使用json表达
#   tag     选填参数。为了方便大家比较多轮的训练，例如在tensorboard中展示多轮曲线。我们会在用户指定的checkpoint文件夹下保存各轮的结果
#           每轮的结果保存在以tag名命名的文件夹内。tag默认为yyyyMMddHHmm格式，用户也可以通过tag参数手动指定
#
####################################################################################################################


#解析脚本Options
MODE="afo"
MODEL_CONF=""
TAG_ID=`/bin/date +"%Y%m%d%H%M%S"`
TIME_STAMP=`/bin/date +"%Y%m%d%H%M%S"`

while [[ $# -gt 0 ]]
    do
        case "$1" in
            --warm_start_id)
            warm_start_id="$2"
            shift
            shift
            ;;
            --mode)
            MODE="$2"
            shift
            shift
            ;;
             --model)
            MODEL_CONF="$2"
            shift
            shift
            ;;
            --tag)
            TAG_ID="$2"
            shift
            shift
            ;;
            --timestamped_saved_model)
            timestamped_saved_model="$2"
            shift
            shift
            ;;
            --eval_file)
            eval_file="$2"
            shift
            shift
            ;;
            --predict_mode)
            predict_mode="$2"
            shift
            shift
            ;;
            --epoch_saved_model_base)
            epoch_saved_model_base="$2"
            shift
            shift
            ;;
           --afo_xml)
            afo_xml="$2"
            shift
            shift
            ;;
            --cv_input_dir)
            cv_input_dir="$2"
            shift
            shift
            ;;
            --cv_fold)
            cv_fold="$2"
            shift
            shift
            ;;
            --test_file)
            test_file="$2"
            shift
            shift
            ;;
            --epoch_samples)
            epoch_samples="$2"
            shift
            shift
            ;;
            --extra_train_file)
            extra_train_file="$2"
            shift
            shift
            ;;
            --extra_train_file_cv)
            extra_train_file_cv="$2"
            shift
            shift
            ;;
            --epochs)
            epochs="$2"
            shift
            shift
            ;;
            
            *)
            
            echo "Unknown option $1. Supported options are [--mode local|afo] [--hparam] [--tag]"
            exit 3
            ;;
        esac
    done

echo "Running in [$MODE] mode with model config file [$MODEL_CONF] and tag [${TAG_ID}]"

function run_local(){
    cd src

    params=()
    #脚本模式为local
    params+=("--script_mode=local")
    #传递run id
    params+=("--run_id=$TAG_ID")
    #传递模型定义文件，本地模式模型定义文件在该脚本同级目录下, 脚本会自动添加 conf/model/
    params+=("--model_conf=conf/model/$MODEL_CONF")
    params+=("--train_train_data=$eval_file")
    params+=("--train_valid_data=$eval_file")

    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"

    python run.py "${params[@]}"
}

function init_env() {
    source /opt/meituan/tensorflow-release/local_env.sh
    source /opt/meituan/hadoop/bin/hadoop_user_login.sh hadoop-mining
}

function generate_tfrecord {
    eixt
}

function generate_ids {
    exit
}

function run_afo(){
    init_env
    afo_base_dir="training.tar.gz"

    params=()
    #脚本模式为local
    params+=("-Dargs.script_mode=afo")
    #传递run id
    params+=("-Dargs.run_id=$TAG_ID")
    #传递模型定义文件，本地模式模型定义文件在该脚本同级目录下, 脚本会自动添加 conf/model/
    params+=("-Dargs.model_conf=${afo_base_dir}/src/conf/model/$MODEL_CONF")

    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"

    if [[ -e /tmp/$TAG_ID/${afo_base_dir} ]]; then
        rm -r /tmp/$TAG_ID
    fi
    mkdir -p /tmp/$TAG_ID && tar -czf /tmp/$TAG_ID/${afo_base_dir} ./src


    mkdir -p archive_trained_model/$TAG_ID
    cp -r ./src archive_trained_model/$TAG_ID

    ${AFO_TF_HOME}/bin/tensorflow-submit.sh \
        -conf src/conf/$afo_xml.xml \
        -files /tmp/$TAG_ID/${afo_base_dir} \
        "${params[@]}"

    rm -r /tmp/$TAG_ID
}

function finish() {
    #清理工作
    echo "bye."
    exit 1
}

function run_local_predict() {
    cd src

    params=()
    #脚本模式为local
    params+=("--script_mode=local")
    #传递run id
    params+=("--run_id=$TAG_ID")
    #传递模型定义文件，本地模式模型定义文件在该脚本同级目录下, 脚本会自动添加 conf/model/
    params+=("--model_conf=conf/model/$MODEL_CONF")

    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"

    valid_pos=../../../data/valid/valid_answer.json

    mkdir -p ../output
    output_file="../output/"prediction_$TAG_ID
    params+=("--output_file=$output_file")
    timestamped_saved_model=${timestamped_saved_model}/
    params+=("--timestamped_saved_model=$timestamped_saved_model")
    params+=("--eval_file=$eval_file")
    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"
    python predict.py "${params[@]}"

    submit_file="../output/"valid_submit_$TAG_ID
    ndcg_of_submit_file="../output/"valid_submit_${TAG_ID}_ndcg
    python eval_ndcg.py $output_file $valid_pos $submit_file all
    python official_eval_ndcg.py $valid_pos $submit_file $ndcg_of_submit_file
    
    
    params=()
    params+=("--timestamped_saved_model=$timestamped_saved_model")
    params+=("--script_mode=local")
    #传递run id
    params+=("--run_id=${TAG_ID}_cv${i}")
    params+=("--model_conf=conf/model/$MODEL_CONF")
    testAFile="$test_file"
    params+=("--eval_file=$testAFile")
    mkdir -p ../output/testA/
    output_file="../output/testA/submit"_${TAG_ID}_$1
    params+=("--output_file=$output_file")
    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"
    python predict.py "${params[@]}"
    
    
    cd ..
}



function run_gen_submit() {
    cd src

    params=()
    #脚本模式为local
    params+=("--script_mode=local")
    #传递run id
    params+=("--run_id=$TAG_ID")
    #传递模型定义文件，本地模式模型定义文件在该脚本同级目录下, 脚本会自动添加 conf/model/
    params+=("--model_conf=conf/model/$MODEL_CONF")

    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"


    mkdir -p ../output
    output_file="../output/"prediction_$TAG_ID
    params+=("--output_file=$output_file")
    timestamped_saved_model=${timestamped_saved_model}/
    params+=("--timestamped_saved_model=$timestamped_saved_model")
    params+=("--eval_file=$eval_file")
    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"
    python predict.py "${params[@]}"

    submit_file="../output/"valid_submit_$TAG_ID
    ndcg_of_submit_file="../output/"valid_submit_${TAG_ID}_ndcg
    python gen_submit.py $output_file $submit_file
    cd ..
}
#catch用户的ctrl-c行为，进行脚本恢复
trap finish SIGINT

function run_epoch_predict() {
    saved_model_base=$epoch_saved_model_base
    timestamp=$(basename $saved_model_base)
    echo $timestamp
    absolute_path=`readlink -f saved_model`
    local_saved_model_folder=${absolute_path}/${timestamp}
    mkdir -p $local_saved_model_folder
    array=$(hadoop fs -ls ${saved_model_base} | awk '{print $NF}' | grep epoch)
    echo $array
    for i in $array; do
        current_epoch=$(basename ${i})
        current_path=$(hadoop fs -ls ${i} | awk '{print $NF}' | grep epoch)
        echo "[INFO] " $current_path
        echo "[INFO] " $current_epoch
        local_epoch_folder=${local_saved_model_folder}/${current_epoch}
        rm -r ${local_epoch_folder} || true
        mkdir -p ${local_epoch_folder}
        hadoop fs -get ${current_path}/* ${local_epoch_folder}
        ls ${local_epoch_folder}
        timestamped_saved_model=${local_epoch_folder}
        run_local_predict $current_epoch
    done
}


function run_last_epoch_predict() {
    saved_model_base=$epoch_saved_model_base
    timestamp=$(basename $saved_model_base)
    echo $timestamp
    absolute_path=`readlink -f saved_model`
    local_saved_model_folder=${absolute_path}/${timestamp}
    mkdir -p $local_saved_model_folder
    array=$(hadoop fs -ls ${saved_model_base} | awk '{print $NF}' | grep epoch)
    echo $array
    
    last_epoch=""
    last_epoch_num=-1
    last_path=""
    for i in $array; do
        current_path=$(hadoop fs -ls ${i} | awk '{print $NF}' | grep epoch)
        current_epoch=$(basename ${i})
        epoch_num=${current_epoch:5}  
        if [[ $epoch_num -gt $last_epoch_num ]]
        then
            last_epoch_num=$epoch_num
            last_epoch=$current_epoch
            last_path=$current_path
        fi
        
        if [ -z "$epoch_num" ]; then
            last_epoch=$current_epoch
            last_path=$current_path
            break
        fi
    done
    
    current_path=$last_path
    current_epoch=$last_epoch
    echo "[INFO] last epoch path " $last_path
    echo "[INFO] last epoch " $last_epoch

   

    local_epoch_folder=${local_saved_model_folder}/${current_epoch}
    rm -r ${local_epoch_folder} || true
    mkdir -p ${local_epoch_folder}
    hadoop fs -get ${current_path}/* ${local_epoch_folder}
    ls ${local_epoch_folder}
    timestamped_saved_model=${local_epoch_folder}
    run_local_predict $current_epoch
}


if [[ "$MODE" = "local" ]];then
    if [[ ! -e src/conf/model/$MODEL_CONF ]];then
        echo "Unable to find model definition json in src/conf/model directory. "
        exit 3
    fi
    run_local
fi

if [[ "$MODE" = "afo" ]];then
    if [[ ! -e src/conf/$afo_xml.xml ]];then
        echo "Unable to find file afo_settings.xml in src/conf/model directory. "
        exit 3
    fi

    run_afo
fi


function local_predict_export(){
    cd src

    params=()
    #脚本模式为local
    params+=("--script_mode=local_predict_export")
    #传递run id
    params+=("--run_id=$TAG_ID")
    params+=("--warm_start_id=${warm_start_id}")
    #传递模型定义文件，本地模式模型定义文件在该脚本同级目录下, 脚本会自动添加 conf/model/
    params+=("--model_conf=conf/model/$MODEL_CONF")
    params+=("--train_train_data=$eval_file")
    params+=("--train_valid_data=$eval_file")

    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"

    python run.py "${params[@]}"
    
    timestamped_saved_model="../../../user_data/export/savedmodel/"$TAG_ID
    timestamped_saved_model=`readlink -f $timestamped_saved_model`
    timestamp=`ls ${timestamped_saved_model}`
    echo "saved timestamp parent folder: " $timestamped_saved_model
    echo "saved timestamp: " $timestamp
    timestamped_saved_model=${timestamped_saved_model}/$timestamp
    params+=("--timestamped_saved_model=$timestamped_saved_model")
    
    
    eval_file="$eval_file"
    params=()
    params+=("--timestamped_saved_model=$timestamped_saved_model")
    params+=("--script_mode=local")
    #传递run id
    params+=("--run_id=${TAG_ID}")
    params+=("--model_conf=conf/model/$MODEL_CONF")
    params+=("--train_train_data=$train_file")
    params+=("--train_valid_data=$eval_file")
    params+=("--eval_file=$eval_file")
    mkdir -p "../output/prediction_vec_"${TAG_ID}
    output_file="../output/prediction_vec_"${TAG_ID}/prediction_${TAG_ID}
    params+=("--output_file=$output_file")
    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"
    python predict_vec.py "${params[@]}"    
    
    params=()
    params+=("--timestamped_saved_model=$timestamped_saved_model")
    params+=("--script_mode=local")
    #传递run id
    params+=("--run_id=${TAG_ID}")
    params+=("--model_conf=conf/model/$MODEL_CONF")
    params+=("--train_train_data=$train_file")
    params+=("--train_valid_data=$eval_file")
    testAFile="$test_file"
    params+=("--eval_file=$testAFile")
    mkdir -p ../output/testA_vec/
    output_file="../output/testA_vec/submit"_${TAG_ID}
    params+=("--output_file=$output_file")
    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"
    python predict_vec.py "${params[@]}"
    
    
}

if [[ "$MODE" = "local_predict_export" ]];then
    if [[ ! -e src/conf/model/$MODEL_CONF ]];then
        echo "Unable to find model definition json in src/conf/model directory. "
        exit 3
    fi
    local_predict_export
fi


if [[ "$MODE" = "local_predict" ]];then
    if [[ ! -e $eval_file ]];then
        echo "Unable to find file $eval_file."
        exit 3
    fi
    if [[ ! -e $timestamped_saved_model ]];then
        echo "Unable to find file $timestamped_saved_model."
        exit 3
    fi
    run_local_predict
fi

if [[ "$MODE" = "epoch_predict" ]];then
    if [[ ! -e $eval_file ]];then
        echo "Unable to find file $eval_file."
        exit 3
    fi
    run_epoch_predict
fi

if [[ "$MODE" = "last_epoch_predict" ]];then
    if [[ ! -e $eval_file ]];then
        echo "Unable to find file $eval_file."
        exit 3
    fi
    run_last_epoch_predict
fi

if [[ "$MODE" = "gen_submit" ]];then
    if [[ ! -e $eval_file ]];then
        echo "Unable to find file $eval_file."
        exit 3
    fi
    run_gen_submit
fi

if [[ "$MODE" = "cv_local" ]] ;then
    for i in $(seq 0 $(expr $cv_fold - 1))
    do
        echo "current cv round is " $i
        train_file="$cv_input_dir/train"${i}".tfrecord"
        predict_file="$cv_input_dir/valid"${i}".tfrecord"
        eval_file="$cv_input_dir/valid"${i}".tfrecord"
        query_ids="$cv_input_dir/valid"${i}"_id.pkl"
        cd src
        params=()
        #脚本模式为local
        params+=("--script_mode=local")
        #传递run id
        params+=("--run_id=${TAG_ID}_cv${i}")
        params+=("--warm_start_id=${warm_start_id}")
        params+=("--epoch_samples=$epoch_samples")
        
        if [ "$epochs" != "" ]; then
            params+=("--epochs=${epochs}")
        fi
        #传递模型定义文件，本地模式模型定义文件在该脚本同级目录下, 脚本会自动添加 conf/model/
        params+=("--model_conf=conf/model/$MODEL_CONF")
        
        if [ -z "$extra_train_file" ]; then
            params+=("--train_train_data=$train_file")
        else
            if [ -z "$extra_train_file_cv" ]; then
                params+=("--train_train_data=$train_file,$extra_train_file")
            else
                params+=("--train_train_data=$train_file,$extra_train_file/train${i}.tfrecord")
            fi
        fi
        params+=("--train_valid_data=$eval_file")
    
        echo "collected ${#params[@]} params."
        printf '%s\n' "${params[@]}"
    
        python run.py "${params[@]}"
    
    
        params+=("--eval_file=$predict_file")
        valid_pos=../../../data/valid/valid_answer.json
        mkdir -p ../output
        mkdir -p "../output/prediction_"${TAG_ID}
        output_file="../output/prediction_"${TAG_ID}/prediction_${TAG_ID}_cv${i}
        params+=("--output_file=$output_file")
        timestamped_saved_model="../../../user_data/export/savedmodel/"${TAG_ID}_cv${i}
        timestamped_saved_model=`readlink -f $timestamped_saved_model`
        timestamp=`ls ${timestamped_saved_model}`
        echo "saved timestamp parent folder: " $timestamped_saved_model
        echo "saved timestamp: " $timestamp
        timestamped_saved_model=${timestamped_saved_model}/$timestamp
        params+=("--timestamped_saved_model=$timestamped_saved_model")
        echo "collected ${#params[@]} params."
        printf '%s\n' "${params[@]}"
        python predict.py "${params[@]}"
        submit_file="../output/"valid_submit_${TAG_ID}_cv${i}
        ndcg_of_submit_file="../output/"valid_submit_${TAG_ID}_cv${i}_ndcg
        python eval_ndcg.py $output_file $valid_pos $submit_file $query_ids
        python official_eval_ndcg.py $valid_pos $submit_file $ndcg_of_submit_file
    
    
    
        params=()
        params+=("--timestamped_saved_model=$timestamped_saved_model")
        params+=("--script_mode=local")
        #传递run id
        params+=("--run_id=${TAG_ID}_cv${i}")
        params+=("--model_conf=conf/model/$MODEL_CONF")
        params+=("--train_train_data=$train_file")
        params+=("--train_valid_data=$eval_file")
        testAFile="$test_file"
        params+=("--eval_file=$testAFile")
        mkdir -p ../output/testA/
        output_file="../output/testA/submit"_${TAG_ID}_cv${i}
        params+=("--output_file=$output_file")
        echo "collected ${#params[@]} params."
        printf '%s\n' "${params[@]}"
        python predict.py "${params[@]}"
        
        
        cd ..
    done
fi



if [[ "$MODE" = "cv_local_predict" ]] ;then
    for i in $(seq 0 $(expr $cv_fold - 1))
    do
        echo "current cv round is " $i
        train_file="$cv_input_dir/train"${i}".tfrecord"
        predict_file="$cv_input_dir/valid"${i}".tfrecord"
        eval_file="$cv_input_dir/valid"${i}".tfrecord"
        query_ids="$cv_input_dir/valid"${i}"_id.pkl"
        cd src
        
        timestamped_saved_model="../../../user_data/export/savedmodel/"${TAG_ID}_cv${i}
        timestamped_saved_model=`readlink -f $timestamped_saved_model`
        timestamp=`ls ${timestamped_saved_model}`
        echo "saved timestamp parent folder: " $timestamped_saved_model
        echo "saved timestamp: " $timestamp
        timestamped_saved_model=${timestamped_saved_model}/$timestamp
        params+=("--timestamped_saved_model=$timestamped_saved_model")
        echo "collected ${#params[@]} params."
        printf '%s\n' "${params[@]}"
        
        
        params=()
        params+=("--timestamped_saved_model=$timestamped_saved_model")
        params+=("--script_mode=local")
        #传递run id
        params+=("--run_id=${TAG_ID}_cv${i}")
        params+=("--model_conf=conf/model/$MODEL_CONF")
        params+=("--train_train_data=$train_file")
        params+=("--train_valid_data=$eval_file")
        predict_file="$test_file"
        params+=("--eval_file=$predict_file")
        mkdir -p ../output/extra/
        test_base_file=`basename ${test_file}`
        output_file="../output/extra/"${test_base_file}_${TAG_ID}_cv${i}
        params+=("--output_file=$output_file")
        echo "collected ${#params[@]} params."
        printf '%s\n' "${params[@]}"
        python predict.py "${params[@]}"
        
        
        cd ..
    done
fi



if [[ "$MODE" = "cv_local_predict_v1" ]] ;then
    for i in $(seq 0 $(expr $cv_fold - 1))
    do
        echo "current cv round is " $i
        train_file="$cv_input_dir/train"${i}".tfrecord"
        predict_file="$cv_input_dir/valid"${i}".tfrecord"
        eval_file="$cv_input_dir/valid"${i}".tfrecord"
        query_ids="$cv_input_dir/valid"${i}"_id.pkl"
        cd src
        
        timestamped_saved_model="../../../user_data/export/savedmodel_v1/"${TAG_ID}_cv${i}
        timestamp=`ls ${timestamped_saved_model}`
        echo "saved timestamp parent folder: " $timestamped_saved_model
        echo "saved timestamp: " $timestamp
        timestamped_saved_model=${timestamped_saved_model}/$timestamp
        
        
        params=()
        params+=("--timestamped_saved_model=$timestamped_saved_model")
        params+=("--script_mode=local")
        #传递run id
        params+=("--run_id=${TAG_ID}_cv${i}")
        params+=("--model_conf=conf/model/$MODEL_CONF")
        params+=("--train_train_data=$train_file")
        params+=("--train_valid_data=$eval_file")
        predict_file="$test_file"
        params+=("--eval_file=$predict_file")
        mkdir -p ../../../user_data/output/
        test_base_file=`basename ${test_file}`
        output_file="../../../user_data/output/"${TAG_ID}_cv${i}
        params+=("--output_file=$output_file")
        echo "collected ${#params[@]} params."
        printf '%s\n' "${params[@]}"
        python predict.py "${params[@]}"
        
        
        cd ..
    done
fi


function local_predict_export_test(){
    cd src

    timestamped_saved_model="../../../user_data/export/savedmodel_v1/"$TAG_ID
    timestamp=`ls ${timestamped_saved_model}`
    echo "saved timestamp parent folder: " $timestamped_saved_model
    echo "saved timestamp: " $timestamp
    timestamped_saved_model=${timestamped_saved_model}/$timestamp
    
    params=()
    params+=("--timestamped_saved_model=$timestamped_saved_model")
    params+=("--script_mode=local")
    #传递run id
    params+=("--run_id=${TAG_ID}")
    params+=("--model_conf=conf/model/$MODEL_CONF")
    params+=("--train_train_data=$train_file")
    params+=("--train_valid_data=$eval_file")
    testAFile="$test_file"
    params+=("--eval_file=$testAFile")
    mkdir -p ../../../user_data/output/testA_vec/
    output_file="../../../user_data/output/testA_vec/submit"_${TAG_ID}
    params+=("--output_file=$output_file")
    echo "collected ${#params[@]} params."
    printf '%s\n' "${params[@]}"
    python predict_vec.py "${params[@]}"
    
}

if [[ "$MODE" = "local_predict_export_test" ]];then
    if [[ ! -e src/conf/model/$MODEL_CONF ]];then
        echo "Unable to find model definition json in src/conf/model directory. "
        exit 3
    fi
    local_predict_export_test
fi
