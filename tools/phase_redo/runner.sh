#!/usr/bin/env bash
# R61 RUNNER: RUNNER <arm_yaml> <output_dir>
# Loads config/ntu_viral.yaml then the arm yaml (merged, not concatenated, so a top-level
# common: block in both files merges instead of colliding), adds ONLY offline_bag_path (eee_01)
# and outputs.path/debug_log_dir -- no other overlay. Exit code = run status.
set -u
ARM_YAML="$1"; OUT_DIR="$2"
REPO=/root/catkin_ws/src/livo_recon
BAG=/root/datasets_test/ntu_viral/eee_01/eee_01.bag
mkdir -p "$OUT_DIR"
MERGED="$OUT_DIR/.merged_overrides.yaml"
python3 - "$ARM_YAML" "$BAG" "$OUT_DIR" "$MERGED" <<'PY'
import sys, yaml
arm_yaml, bag, out_dir, merged_path = sys.argv[1:5]
with open(arm_yaml) as f:
    doc = yaml.safe_load(f) or {}
def deepset(d, path, value):
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value
deepset(doc, ["common", "offline_bag_path"], bag)
deepset(doc, ["outputs", "path"], out_dir)
deepset(doc, ["outputs", "debug_log_dir"], out_dir)
with open(merged_path, "w") as f:
    yaml.safe_dump(doc, f, default_flow_style=False)
PY
PORT="${R61_PORT:-11340}"
export ROS_MASTER_URI="http://localhost:${PORT}"
export ROS_HOSTNAME=localhost
set +u
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
set -u
cd /root/catkin_ws
yes | rosnode cleanup > /dev/null 2>&1 || true
rosparam delete / > /dev/null 2>&1 || true
roslaunch livo_recon livo_recon_ntu_viral.launch overrides_file:="$MERGED" rviz:=false > "$OUT_DIR/run.log" 2>&1
rc=$?
# Match run_job.sh's own "last well-formed [evo line" convention so results_lio.txt stays in the
# same "one line, contains ATE=" shape every downstream reader expects.
ATE_LINE_RE='ATE=[0-9.eE+-]+m'
last_evo="$(grep -E '\[evo.*'"$ATE_LINE_RE" "$OUT_DIR/run.log" 2>/dev/null | tail -1)"
[[ "$last_evo" =~ $ATE_LINE_RE ]] && echo "$last_evo" > "$OUT_DIR/results_lio.txt"
exit $rc
