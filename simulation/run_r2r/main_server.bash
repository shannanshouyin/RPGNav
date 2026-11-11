export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

PRETRAIN_CKPT=${PRETRAIN_CKPT:-pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt}
EVAL_CKPT=${3:-data/logs/checkpoints/release_r2r_dagger/ckpt.iter3800.pth}


flag1="--exp_name release_r2r_dagger
      --run-type dagger
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.DATASET.SUFFIX _90
      MODEL.pretrained_path ${PRETRAIN_CKPT}
      "

flag2=" --exp_name release_r2r_dagger
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR ${EVAL_CKPT}
      IL.back_algo control
      MODEL.pretrained_path ${PRETRAIN_CKPT}
      "

flag_zero_shot=" --exp_name release_r2r_zeroshot
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.back_algo control
      MODEL.pretrained_path ${PRETRAIN_CKPT}
      EVAL.ZERO_SHOT_PRETRAIN True
      "

mode=$1
if [[ "$mode" == "eval" || "$mode" == "infer" ]]; then
      EXTRA_OPTS="${@:4}"
else
      EXTRA_OPTS="${@:3}"
fi
case $mode in 
      dagger)
      echo "###### dagger train mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag1 $EXTRA_OPTS
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag2 $EXTRA_OPTS
      ;;
      zero_shot|zeroshot)
      echo "###### zero-shot eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag_zero_shot $EXTRA_OPTS
      ;;
esac

# 命令行运行：
# bash run_r2r/main_server.bash dagger 2333
