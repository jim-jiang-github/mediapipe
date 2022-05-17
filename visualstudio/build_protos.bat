%cd%/protoc.exe --proto_path=../third_party/protobuf/src --proto_path=../third_party/tensorflow --proto_path=../ --cpp_out=./ ^
mediapipe/calculators/audio/mfcc_mel_calculators.proto ^
mediapipe/calculators/audio/rational_factor_resample_calculator.proto ^
mediapipe/calculators/audio/spectrogram_calculator.proto ^
mediapipe/calculators/audio/stabilized_log_calculator.proto ^
mediapipe/calculators/audio/time_series_framer_calculator.proto ^
mediapipe/calculators/core/clip_vector_size_calculator.proto ^
mediapipe/calculators/core/concatenate_vector_calculator.proto ^
mediapipe/calculators/core/constant_side_packet_calculator.proto ^
mediapipe/calculators/core/dequantize_byte_array_calculator.proto ^
mediapipe/calculators/core/flow_limiter_calculator.proto ^
mediapipe/calculators/core/gate_calculator.proto ^
mediapipe/calculators/core/graph_profile_calculator.proto ^
mediapipe/calculators/core/packet_cloner_calculator.proto ^
mediapipe/calculators/core/packet_resampler_calculator.proto ^
mediapipe/calculators/core/packet_thinner_calculator.proto ^
mediapipe/calculators/core/quantize_float_vector_calculator.proto ^
mediapipe/calculators/core/sequence_shift_calculator.proto ^
mediapipe/calculators/core/split_vector_calculator.proto  ^
mediapipe/calculators/image/bilateral_filter_calculator.proto ^
mediapipe/calculators/image/feature_detector_calculator.proto ^
mediapipe/calculators/image/image_clone_calculator.proto ^
mediapipe/calculators/image/image_cropping_calculator.proto ^
mediapipe/calculators/image/image_transformation_calculator.proto ^
mediapipe/calculators/image/mask_overlay_calculator.proto ^
mediapipe/calculators/image/opencv_encoded_image_to_image_frame_calculator.proto ^
mediapipe/calculators/image/opencv_image_encoder_calculator.proto ^
mediapipe/calculators/image/recolor_calculator.proto ^
mediapipe/calculators/image/scale_image_calculator.proto ^
mediapipe/calculators/image/segmentation_smoothing_calculator.proto ^
mediapipe/calculators/image/set_alpha_calculator.proto ^
mediapipe/calculators/image/warp_affine_calculator.proto ^
mediapipe/calculators/internal/callback_packet_calculator.proto ^
mediapipe/calculators/tensor/image_to_tensor_calculator.proto ^
mediapipe/calculators/tensor/inference_calculator.proto ^
mediapipe/calculators/tensor/landmarks_to_tensor_calculator.proto ^
mediapipe/calculators/tensor/tensor_converter_calculator.proto ^
mediapipe/calculators/tensor/tensors_to_classification_calculator.proto ^
mediapipe/calculators/tensor/tensors_to_detections_calculator.proto ^
mediapipe/calculators/tensor/tensors_to_floats_calculator.proto ^
mediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto ^
mediapipe/calculators/tensor/tensors_to_segmentation_calculator.proto ^
mediapipe/calculators/tensorflow/graph_tensors_packet_generator.proto ^
mediapipe/calculators/tensorflow/image_frame_to_tensor_calculator.proto ^
mediapipe/calculators/tensorflow/lapped_tensor_buffer_calculator.proto ^
mediapipe/calculators/tensorflow/matrix_to_tensor_calculator_options.proto ^
mediapipe/calculators/tensorflow/object_detection_tensors_to_detections_calculator.proto ^
mediapipe/calculators/tensorflow/pack_media_sequence_calculator.proto ^
mediapipe/calculators/tensorflow/tensor_squeeze_dimensions_calculator.proto ^
mediapipe/calculators/tensorflow/tensor_to_image_frame_calculator.proto ^
mediapipe/calculators/tensorflow/tensor_to_matrix_calculator.proto ^
mediapipe/calculators/tensorflow/tensor_to_vector_float_calculator_options.proto ^
mediapipe/calculators/tensorflow/tensor_to_vector_string_calculator_options.proto ^
mediapipe/calculators/tensorflow/tensorflow_inference_calculator.proto ^
mediapipe/calculators/tensorflow/tensorflow_session_from_frozen_graph_calculator.proto ^
mediapipe/calculators/tensorflow/tensorflow_session_from_frozen_graph_generator.proto ^
mediapipe/calculators/tensorflow/tensorflow_session_from_saved_model_calculator.proto ^
mediapipe/calculators/tensorflow/tensorflow_session_from_saved_model_generator.proto ^
mediapipe/calculators/tensorflow/unpack_media_sequence_calculator.proto ^
mediapipe/calculators/tensorflow/vector_float_to_tensor_calculator_options.proto ^
mediapipe/calculators/tensorflow/vector_int_to_tensor_calculator_options.proto ^
mediapipe/calculators/tensorflow/vector_string_to_tensor_calculator_options.proto ^
mediapipe/calculators/tflite/ssd_anchors_calculator.proto ^
mediapipe/calculators/tflite/tflite_converter_calculator.proto ^
mediapipe/calculators/tflite/tflite_custom_op_resolver_calculator.proto ^
mediapipe/calculators/tflite/tflite_inference_calculator.proto ^
mediapipe/calculators/tflite/tflite_tensors_to_classification_calculator.proto ^
mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto ^
mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.proto ^
mediapipe/calculators/tflite/tflite_tensors_to_segmentation_calculator.proto ^
mediapipe/calculators/util/annotation_overlay_calculator.proto ^
mediapipe/calculators/util/association_calculator.proto ^
mediapipe/calculators/util/collection_has_min_size_calculator.proto ^
mediapipe/calculators/util/detection_label_id_to_text_calculator.proto ^
mediapipe/calculators/util/detections_to_rects_calculator.proto ^
mediapipe/calculators/util/detections_to_render_data_calculator.proto ^
mediapipe/calculators/util/filter_detections_calculator.proto ^
mediapipe/calculators/util/labels_to_render_data_calculator.proto ^
mediapipe/calculators/util/landmark_projection_calculator.proto ^
mediapipe/calculators/util/landmarks_refinement_calculator.proto ^
mediapipe/calculators/util/landmarks_smoothing_calculator.proto ^
mediapipe/calculators/util/landmarks_to_detection_calculator.proto ^
mediapipe/calculators/util/landmarks_to_floats_calculator.proto ^
mediapipe/calculators/util/landmarks_to_render_data_calculator.proto ^
mediapipe/calculators/util/latency.proto ^
mediapipe/calculators/util/local_file_contents_calculator.proto ^
mediapipe/calculators/util/logic_calculator.proto ^
mediapipe/calculators/util/non_max_suppression_calculator.proto ^
mediapipe/calculators/util/packet_frequency.proto ^
mediapipe/calculators/util/packet_frequency_calculator.proto ^
mediapipe/calculators/util/packet_latency_calculator.proto ^
mediapipe/calculators/util/rect_to_render_data_calculator.proto ^
mediapipe/calculators/util/rect_to_render_scale_calculator.proto ^
mediapipe/calculators/util/rect_transformation_calculator.proto ^
mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto ^
mediapipe/calculators/util/thresholding_calculator.proto ^
mediapipe/calculators/util/timed_box_list_id_to_label_calculator.proto ^
mediapipe/calculators/util/timed_box_list_to_render_data_calculator.proto ^
mediapipe/calculators/util/top_k_scores_calculator.proto ^
mediapipe/calculators/util/visibility_copy_calculator.proto ^
mediapipe/calculators/util/visibility_smoothing_calculator.proto ^
mediapipe/calculators/video/box_detector_calculator.proto ^
mediapipe/calculators/video/box_tracker_calculator.proto ^
mediapipe/calculators/video/flow_packager_calculator.proto ^
mediapipe/calculators/video/flow_to_image_calculator.proto ^
mediapipe/calculators/video/motion_analysis_calculator.proto ^
mediapipe/calculators/video/opencv_video_encoder_calculator.proto ^
mediapipe/calculators/video/tracked_detection_manager_calculator.proto ^
mediapipe/calculators/video/video_pre_stream_calculator.proto ^
mediapipe/framework/calculator.proto ^
mediapipe/framework/calculator_options.proto ^
mediapipe/framework/calculator_profile.proto ^
mediapipe/framework/mediapipe_options.proto ^
mediapipe/framework/packet_factory.proto ^
mediapipe/framework/packet_generator.proto ^
mediapipe/framework/status_handler.proto ^
mediapipe/framework/stream_handler.proto ^
mediapipe/framework/thread_pool_executor.proto ^
mediapipe/framework/deps/proto_descriptor.proto ^
mediapipe/framework/formats/affine_transform_data.proto ^
mediapipe/framework/formats/body_rig.proto ^
mediapipe/framework/formats/classification.proto ^
mediapipe/framework/formats/detection.proto ^
mediapipe/framework/formats/image_file_properties.proto ^
mediapipe/framework/formats/image_format.proto ^
mediapipe/framework/formats/landmark.proto ^
mediapipe/framework/formats/location_data.proto ^
mediapipe/framework/formats/matrix_data.proto ^
mediapipe/framework/formats/rect.proto ^
mediapipe/framework/formats/time_series_header.proto ^
mediapipe/framework/formats/annotation/locus.proto ^
mediapipe/framework/formats/annotation/rasterization.proto ^
mediapipe/framework/formats/motion/optical_flow_field_data.proto ^
mediapipe/framework/formats/object_detection/anchor.proto ^
mediapipe/framework/stream_handler/default_input_stream_handler.proto ^
mediapipe/framework/stream_handler/fixed_size_input_stream_handler.proto ^
mediapipe/framework/stream_handler/sync_set_input_stream_handler.proto ^
mediapipe/framework/stream_handler/timestamp_align_input_stream_handler.proto ^
mediapipe/framework/tool/calculator_graph_template.proto ^
mediapipe/framework/tool/field_data.proto ^
mediapipe/framework/tool/node_chain_subgraph.proto ^
mediapipe/framework/tool/packet_generator_wrapper_calculator.proto ^
mediapipe/framework/tool/source.proto ^
mediapipe/framework/tool/switch_container.proto ^
mediapipe/gpu/copy_calculator.proto ^
mediapipe/gpu/gl_context_options.proto ^
mediapipe/gpu/gl_scaler_calculator.proto ^
mediapipe/gpu/gl_surface_sink_calculator.proto ^
mediapipe/gpu/gpu_origin.proto ^
mediapipe/gpu/scale_mode.proto ^
mediapipe/graph/instant_motion_tracking/calculators/sticker_buffer.proto ^
mediapipe/graph/iris_tracking/calculators/iris_to_depth_calculator.proto ^
mediapipe/graph/iris_tracking/calculators/iris_to_render_data_calculator.proto ^
mediapipe/graph/object_detection_3d/calculators/annotations_to_model_matrices_calculator.proto ^
mediapipe/graph/object_detection_3d/calculators/annotations_to_render_data_calculator.proto ^
mediapipe/graph/object_detection_3d/calculators/gl_animation_overlay_calculator.proto ^
mediapipe/graph/object_detection_3d/calculators/model_matrix.proto ^
mediapipe/modules/face_geometry/effect_renderer_calculator.proto ^
mediapipe/modules/face_geometry/env_generator_calculator.proto ^
mediapipe/modules/face_geometry/geometry_pipeline_calculator.proto ^
mediapipe/modules/face_geometry/protos/environment.proto ^
mediapipe/modules/face_geometry/protos/face_geometry.proto ^
mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.proto ^
mediapipe/modules/face_geometry/protos/mesh_3d.proto ^
mediapipe/modules/holistic_landmark/calculators/roi_tracking_calculator.proto ^
mediapipe/modules/objectron/calculators/a_r_capture_metadata.proto ^
mediapipe/modules/objectron/calculators/annotation_data.proto ^
mediapipe/modules/objectron/calculators/belief_decoder_config.proto ^
mediapipe/modules/objectron/calculators/camera_parameters.proto ^
mediapipe/modules/objectron/calculators/filter_detection_calculator.proto ^
mediapipe/modules/objectron/calculators/frame_annotation_to_rect_calculator.proto ^
mediapipe/modules/objectron/calculators/frame_annotation_tracker_calculator.proto ^
mediapipe/modules/objectron/calculators/lift_2d_frame_annotation_to_3d_calculator.proto ^
mediapipe/modules/objectron/calculators/object.proto ^
mediapipe/modules/objectron/calculators/tensors_to_objects_calculator.proto ^
mediapipe/modules/objectron/calculators/tflite_tensors_to_objects_calculator.proto ^
mediapipe/util/audio_decoder.proto ^
mediapipe/util/color.proto ^
mediapipe/util/render_data.proto ^
mediapipe/util/tracking/box_detector.proto ^
mediapipe/util/tracking/box_tracker.proto ^
mediapipe/util/tracking/camera_motion.proto ^
mediapipe/util/tracking/flow_packager.proto ^
mediapipe/util/tracking/frame_selection.proto ^
mediapipe/util/tracking/frame_selection_solution_evaluator.proto ^
mediapipe/util/tracking/motion_analysis.proto ^
mediapipe/util/tracking/motion_estimation.proto ^
mediapipe/util/tracking/motion_models.proto ^
mediapipe/util/tracking/motion_saliency.proto ^
mediapipe/util/tracking/push_pull_filtering.proto ^
mediapipe/util/tracking/region_flow.proto ^
mediapipe/util/tracking/region_flow_computation.proto ^
mediapipe/util/tracking/tone_estimation.proto ^
mediapipe/util/tracking/tone_models.proto ^
mediapipe/util/tracking/tracked_detection_manager_config.proto ^
mediapipe/util/tracking/tracking.proto

@echo off
timeout /t 15