

CEDAR_CKPT_DIR=/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/checkpoints
PACE_CKPT_DIR=/storage/home/hcoda1/2/zchen927/p-dxu345-0/openpi/checkpoints


CONFIG_NAME=pi05_libero_cotraining_SRBC
CKPT_NAME=liberogoal_pi_interpolation1122_SRBC_0111
EPOCH=1999

# CONFIG_NAME=pi05_libero_cotraining_interpolation
# CKPT_NAME=liberogoal_pi_interpolation_MI_sampled_0109
# EPOCH=1999

mkdir -p $CEDAR_CKPT_DIR/$CONFIG_NAME/$CKPT_NAME/$EPOCH/
rsync -avz $PACE_CKPT_DIR/$CONFIG_NAME/$CKPT_NAME/$EPOCH/ $CEDAR_CKPT_DIR/$CONFIG_NAME/$CKPT_NAME/$EPOCH/