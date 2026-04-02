#!/bin/bash
# Deploy script: copies source files to their destination locations
# Run this instead of relying on symbolic links

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Deploying files from $SCRIPT_DIR to project locations..."

# Remove existing symlinks/files and copy fresh versions
rm -f "$PROJECT_ROOT/human_point_follower_start.py"
cp "$SCRIPT_DIR/human_point_follower_start.py" "$PROJECT_ROOT/human_point_follower_start.py"
echo "  Copied human_point_follower_start.py -> $PROJECT_ROOT/"

rm -f "$PROJECT_ROOT/human_point_follower_table_demo_start.py"
cp "$SCRIPT_DIR/human_point_follower_table_demo_start.py" "$PROJECT_ROOT/human_point_follower_table_demo_start.py"
echo "  Copied human_point_follower_table_demo_start.py -> $PROJECT_ROOT/"

rm -f "$PROJECT_ROOT/src/hsr-omniverse/human_arm_controller.py"
cp "$SCRIPT_DIR/human_arm_controller.py" "$PROJECT_ROOT/src/hsr-omniverse/human_arm_controller.py"
echo "  Copied human_arm_controller.py -> $PROJECT_ROOT/src/hsr-omniverse/"

rm -f "$PROJECT_ROOT/src/hsr-omniverse/human_point_follower_world.py"
cp "$SCRIPT_DIR/human_point_follower_world.py" "$PROJECT_ROOT/src/hsr-omniverse/human_point_follower_world.py"
echo "  Copied human_point_follower_world.py -> $PROJECT_ROOT/src/hsr-omniverse/"

rm -f "$PROJECT_ROOT/src/hsr-omniverse/table_demo_world.py"
cp "$SCRIPT_DIR/table_demo_world.py" "$PROJECT_ROOT/src/hsr-omniverse/table_demo_world.py"
echo "  Copied table_demo_world.py -> $PROJECT_ROOT/src/hsr-omniverse/"

echo "Done."
