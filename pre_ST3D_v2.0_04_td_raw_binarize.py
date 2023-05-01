
import cv2
import sys
import pickle

from AccessMath.util.opencv_video_player import OpenCVVideoPlayer
from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess

from AccessMath.preprocessing.content.MLBinarizer import MLBinarizer

def process_input(process, input_data):
    # detection_results = input_data[0]

    min_td_confidence = Parameters.TDStab_min_confidence        # 0.65
    use_ML_binarization = Parameters.TDBin_ML_binarization

    # create a video player ...
    m_videos = [video["path"] for video in process.current_lecture.main_videos]
    if "forced_width" in process.current_lecture.parameters:
        forced_res = (process.current_lecture.parameters["forced_width"],
                      process.current_lecture.parameters["forced_height"])
    else:
        forced_res = None

    if use_ML_binarization:
        print("... Loading machine learning binarizer ...")
        in_file = open(Parameters.MLBin_classifier_file, "rb")
        classifier = pickle.load(in_file)
        in_file.close()
    else:
        classifier = None

    ml_binarizer = MLBinarizer(classifier, Parameters.MLBin_patch_size, Parameters.MLBin_sigma_color,
                               Parameters.MLBin_sigma_space, Parameters.MLBin_median_blur_k,
                               Parameters.MLBin_dark_background)

    # TODO: replace this here with the image-based sampling ....
    # open the video ....
    video_player = OpenCVVideoPlayer(m_videos, forced_res)
    video_player.play()

    # sorted key-frames
    keyframe_ids = sorted(list(detection_results.keys()))

    frame_times = []
    frame_indices = []
    compressed_frames = []

    # ... extract and binarize frames using text-detection results ...
    for idx, keyframe_idx in enumerate(keyframe_ids):
        print("-> Binarizing Frame #{0:d} ({1:d} of {2:d})".format(keyframe_idx, idx + 1, len(keyframe_ids)), end="\r")

        text_boxes = detection_results[keyframe_idx]["bboxes"]
        confidences = detection_results[keyframe_idx]["confidences"]

        # get corresponding frame ...
        video_player.set_position_frame(keyframe_idx, False)
        frame_img, frame_idx = video_player.get_frame()

        # binarize ...
        if use_ML_binarization:
            binary = ml_binarizer.binarize_bboxes(frame_img, text_boxes, classifier, confidences, min_td_confidence)
        else:
            binary = ml_binarizer.binarize_bboxes_OTSU(frame_img, text_boxes, confidences, min_td_confidence)

        # cv2.imwrite("output/images/{0:d}.png".format(keyframe_idx), binary)

        # compress ...
        flag, compressed_binary = cv2.imencode(".png", binary)
        # ... add to final results ...
        frame_times.append(detection_results[keyframe_idx]["abs_time"])
        frame_indices.append(keyframe_idx)
        compressed_frames.append(compressed_binary)

    print("\n-> Binarization complete!")
    return frame_times, frame_indices, compressed_frames


def main():
    sys.argv = ['pre_ST3D_v2.0_04_td_raw_binarize.py', \
        'test_data/databases/db_AccessMath2015.xml', '-d', 'testing']
    print('sys.argv', sys.argv)

    # usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    # TODO: read this from parameters:
    input_prefix = Parameters.Output_TextDetection
    output_prefix = Parameters.Output_Binarize
    # process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], input_prefix, output_prefix)
    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], None, output_prefix)

    if not process.initialize():
       return

    process.start_input_processing(process_input)

    print("Finished")


if __name__ == "__main__":
    main()
