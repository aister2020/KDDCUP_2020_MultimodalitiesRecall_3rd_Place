echo "run_cv"


while [[ $# -gt 0 ]]
    do
        case "$1" in
           
            --prefix)
            prefix="$2"
            shift
            shift
            ;;
            --predict_file)
            predict_file="$2"
            shift
            shift
            ;;
            --tag)
            TAG_ID="$2"
            shift
            shift
            ;;
            *)
            
            echo "Unknown option $1. Supported options are [--mode local|afo] [--hparam] [--tag]"
            exit 3
            ;;
        esac
    done


for i in 1
    do
    echo "current cv round is " $i
    
    ./run.sh --mode cv_local_predict_v1 --model local_att_pointwise.json --cv_input_dir ../../../user_data/cv_valid$i --test_file $predict_file --cv_fold 5 --tag ${TAG_ID}_${i}
    done
