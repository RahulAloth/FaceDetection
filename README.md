# Prerequistics
PYthon Knowlege and OpenCV Knowlege.

# 1) Template Matching
python template_matching.py --image path/to/scene.jpg --template path/to/template.png --show
# Optional: find multiple matches with simple NMS and save result
python template_matching.py --image scene.jpg --template patch.png --threshold 0.85 --nms-iou 0.3 --save-out matched_side_by_side.png

# 2) Eye Detection with Haar Cascades
python eye_detection_haar.py --image path/to/faces.jpg --show
# Optional: tweak parameters and save output
python eye_detection_haar.py --image faces.jpg --scale-factor 1.05 --min-neighbors 18 --min-size 12,12 --save-out eyes_out.png
``
