// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
  this is a tflite OpResolver module for loading 'face_landmark_with_attention.tflite'.

  re-written from 3 files -
    ./mediapipe/mediapipe/util/tflite/operations/landmarks_to_transform_matrix.cc
    ./mediapipe/mediapipe/util/tflite/operations/transform_landmarks.cc
    ./mediapipe/mediapipe/util/tflite/operations/transform_tensor_bilinear.cc

  -andre.hl.chen@gmail.com 2021/12/10
*/

#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "flatbuffers/flexbuffers.h"

namespace {

//
// to load 'face_landmark_with_attention.tflite', it need to register 3 operations:
//  1) TransformTensorBilinear
//  2) TransformLandmarks
//  3) Landmarks2TransformMatrix
//

// TransformTensorBilinear
TfLiteStatus TransformTensorBilinear_Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
  TfLiteTensor const* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=input);
  TfLiteTensor const* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=output);

  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  return kTfLiteOk;
}
TfLiteStatus TransformTensorBilinear_Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor const* input0 = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=input0);
  TfLiteTensor const* input1 = tflite::GetInput(context, node, 1);
  TF_LITE_ENSURE(context, nullptr!=input1);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=output);

  tflite::RuntimeShape const& input0_shape = tflite::GetTensorShape(input0);
  float const* const input_data = tflite::GetTensorData<float>(input0);
  tflite::RuntimeShape const& input1_shape = tflite::GetTensorShape(input1);
  tflite::RuntimeShape const& output_shape = tflite::GetTensorShape(output);
  TFLITE_CHECK_EQ(input0_shape.DimensionsCount(), 4);
  TFLITE_CHECK_EQ(output_shape.DimensionsCount(), 4);

  // a 4x4 transformmation matrix
  float const* const transform_matrix = tflite::GetTensorData<float>(input1);
  // Align corners correction: T -> S * ( T * A ), where T is a
  // transformation matrix, and subtruction and addition matrices are:
  // S            A
  // 1 0 0 -0.5   1 0 0 0.5
  // 0 1 0 -0.5   0 1 0 0.5
  // 0 0 1 0      0 0 1 0
  // 0 0 0 1      0 0 0 1
  // Transformation matrix column 3 and rows 3, 4 are identity, which makes
  // the final formula pretty simple and easy to get if doing a manual
  // multiuplication.
  float const m11 = transform_matrix[0];
  float const m12 = transform_matrix[1];
  float const m13 = transform_matrix[3] + m11*0.5f + m12*0.5f - 0.5f;
  float const m21 = transform_matrix[4];
  float const m22 = transform_matrix[5];
  float const m23 = transform_matrix[7] + m21*0.5f + m22*0.5f - 0.5f;

  // 16hx16wx32c
  int const output_height = output_shape.Dims(1);
  int const output_width = output_shape.Dims(2);
  int const output_channels = output_shape.Dims(3);

  // 48hx48wx32c
  int const input_height = input0_shape.Dims(1);
  int const input_width = input0_shape.Dims(2);
  int const input_channels = input0_shape.Dims(3);
  int const input_stride = input_width*input_channels;
  TFLITE_CHECK_EQ(input_channels, output_channels);

  float const tx_max = (float) (input_width - 1);
  float const ty_max = (float) (input_height - 1);
  float tx, ty, frac, inv_frac, v0, v1;
  int ix, iy;

  float* output_data = tflite::GetTensorData<float>(output);
  //memset(output_data, 0, output_height*output_width*output_channels*sizeof(float));
  for (int h=0; h<output_height; ++h) {
    for (int w=0; w<output_width; ++w) {
      tx = m11*(float)w + m12*(float)h + m13;
      ty = m21*(float)w + m22*(float)h + m23;
      if (0.0f<=tx && tx<=tx_max && 0.0f<=ty && ty<=ty_max) {
        for (int c=0; c<output_channels; ++c) {
          ix = (int) floor(tx);
          iy = (int) floor(ty);
          float const* p = input_data + (iy*input_width+ix)*input_channels + c;
          if ((ix+1)<input_width) {
            frac = tx - (float)ix;
            if ((iy+1)<input_height) {
              inv_frac = 1.0f - frac;
              v0 = inv_frac*p[0] + frac*p[input_channels];
              v1 = inv_frac*p[input_stride] + frac*p[input_stride+input_channels];

              frac = ty - (float) iy;
              *output_data++ = (1.0f-frac)*v0 + frac*v1;
            } else {
              *output_data++ = (1.0f-frac)*p[0] + frac*p[input_channels];
            }
          } else if ((iy+1)<input_height) {
            frac = ty - (float) iy;
            *output_data++ = (1.0f-frac)*p[0] + frac*p[input_stride];
          } else {
            *output_data++ = p[0];
          }
        }
      } else {
        memset(output_data, 0, output_channels*sizeof(float));
        output_data += output_channels;
      }
    }
  }
  return kTfLiteOk;
}

// TransformLandmarks
TfLiteStatus TransformLandmarks_Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
  TfLiteTensor const* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=input);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=output);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 3);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  tflite::RuntimeShape const& output_shape = tflite::GetTensorShape(output);
  auto const& input_dim = input->dims->data;
  if (3!=output_shape.DimensionsCount() ||
      input_dim[0]!=output_shape.Dims(0) ||
      input_dim[1]!=output_shape.Dims(1) ||
      input_dim[2]!=output_shape.Dims(2)) {
    // output tensor will take the ownership, you can't deletet the pointer here
    TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
    output_size->data[0] = input_dim[0];
    output_size->data[1] = input_dim[1];
    output_size->data[2] = input_dim[2];
    return context->ResizeTensor(context, output, output_size);
  }
  return kTfLiteOk;
}
TfLiteStatus TransformLandmarks_Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=output);
  TfLiteTensor const* input0 = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=input0);
  TfLiteTensor const* input1 = tflite::GetInput(context, node, 1);
  TF_LITE_ENSURE(context, nullptr!=input1);

  tflite::RuntimeShape const& input0_shape = tflite::GetTensorShape(input0);
  tflite::RuntimeShape const& output_shape = tflite::GetTensorShape(output);
  int const num_landmarks = output_shape.Dims(1);
  TFLITE_CHECK_EQ(input0_shape.DimensionsCount(), 3);
  TFLITE_CHECK_EQ(output_shape.DimensionsCount(), 3);
  TFLITE_CHECK_EQ(input0_shape.Dims(1), num_landmarks);

  // input landmarks array
  float const* landmarks = tflite::GetTensorData<float>(input0);
  int const input_dimensions = input0_shape.Dims(2);
  TFLITE_CHECK_GE(input_dimensions, 2);

  // a 4x4 transformation matrix
  float const* transform_matrix = tflite::GetTensorData<float>(input1);
  float const m11 = transform_matrix[0];
  float const m12 = transform_matrix[1];
  float const m13 = transform_matrix[3];
  float const m21 = transform_matrix[4];
  float const m22 = transform_matrix[5];
  float const m23 = transform_matrix[7];
  float x, y;

  // output
  float* output_data = tflite::GetTensorData<float>(output);
  int const output_dimensions = output_shape.Dims(2);
  if (2==output_dimensions) {
    for (int i=0; i<num_landmarks; ++i,landmarks+=input_dimensions) {
      x = landmarks[0];
      y = landmarks[1];
      *output_data++ = m11*x + m12*y + m13;
      *output_data++ = m21*x + m22*y + m23;
    }
  } else if (3==output_dimensions) {
    for (int i=0; i<num_landmarks; ++i,landmarks+=input_dimensions) {
      x = landmarks[0];
      y = landmarks[1];
      *output_data++ = m11*x + m12*y + m13;
      *output_data++ = m21*x + m22*y + m23;
      *output_data++ = landmarks[2];
    }
  } else {
    context->ReportError(context, "Incorrect output dimensions size: %d", output_dimensions);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

// Landmarks2TransformMatrix
TfLiteStatus LandmarksToTransformMatrix_Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
  TfLiteTensor const* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=input);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=output);

  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 3);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  // a 4x4 matrix shape (1,4,4)
  tflite::RuntimeShape const& output_shape = tflite::GetTensorShape(output);
  if (3!=output_shape.DimensionsCount() ||
      1!=output_shape.Dims(0) || 4!=output_shape.Dims(1) || 4!=output_shape.Dims(2)) {
    // output tensor will take the ownership, you can't deletet the pointer here
    TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
    output_size->data[0] = 1;
    output_size->data[1] = 4;
    output_size->data[2] = 4;
    return context->ResizeTensor(context, output, output_size);
  }
  return kTfLiteOk;
}
TfLiteStatus LandmarksToTransformMatrix_Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=output);
  //tflite::RuntimeShape const& output_shape = tflite::GetTensorShape(output);
  //TF_LITE_ENSURE_EQ(context, output_shape.Dims(0), 1);
  //TF_LITE_ENSURE_EQ(context, output_shape.Dims(1), 4);
  //TF_LITE_ENSURE_EQ(context, output_shape.Dims(2), 4);

  TfLiteTensor const* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, nullptr!=input);
  tflite::RuntimeShape const& input0_shape = tflite::GetTensorShape(input);
  int const num_landmarks = input0_shape.Dims(1); // 468
  int const dim_landmarks = input0_shape.Dims(2);
  TF_LITE_ENSURE(context, num_landmarks>0);
  TF_LITE_ENSURE_EQ(context, dim_landmarks, 3);
  float const* const landmarks = tflite::GetTensorData<float>(input);

  flexbuffers::Map const m =
      flexbuffers::GetRoot((uint8_t const*)(node->custom_initial_data),
                           node->custom_initial_data_size).AsMap();

  float cos_theta = 1.0f;
  float sin_theta = 0.0f;
  {
    int const left_rotation_idx = m["left_rotation_idx"].AsInt32();
    int const right_rotation_idx = m["right_rotation_idx"].AsInt32();
    if (left_rotation_idx>=0 && right_rotation_idx>=0) {
      float const* p1 = landmarks + left_rotation_idx*dim_landmarks;
      float const* p2 = landmarks + right_rotation_idx*dim_landmarks;
      sin_theta = m["target_rotation_radians"].AsFloat() - std::atan2(p2[1]-p1[1], p2[0]-p1[0]);
      cos_theta = std::cos(sin_theta);
      sin_theta = std::sin(sin_theta);
    } else {
      context->ReportError(context, "Incorrect rotation_idx: %d/%d", left_rotation_idx, right_rotation_idx);
      return kTfLiteError;
    }
  }

  float scale_x = m["scale_x"].AsFloat();
  float scale_y = m["scale_y"].AsFloat();
  float crop_x(0.0f), crop_y(0.0f), sx_bx(0.0f), sy_by(0.0f);
  if (scale_x>0.0f && scale_y>0.0f) {
    float const output_height = (float) m["output_height"].AsInt32();
    float const output_width = (float) m["output_width"].AsInt32();
    if (output_width>0.0f && output_height>0.0f) {
      flexbuffers::TypedVector const subset_indices = m["subset_idxs"].AsTypedVector();
      int const total_indices = (int) subset_indices.size();
      int valid_indices(0), id;
      float x_inf(1.e+20f), y_inf(1.e+20f), x_sup(-1.e+20f), y_sup(-1.e+20f), x, y;
      for (int i=0; i<total_indices; ++i) {
        id = subset_indices[i].AsInt32();
        if (id>=0) {
          ++valid_indices;
          auto const* landmark = landmarks + id*dim_landmarks;
          x = landmark[0];
          y = landmark[1];
          crop_x = cos_theta*x - sin_theta*y;
          crop_y = sin_theta*x + cos_theta*y;
          if (x_sup<crop_x) x_sup = crop_x;
          if (y_sup<crop_y) y_sup = crop_y;
          if (x_inf>crop_x) x_inf = crop_x;
          if (y_inf>crop_y) y_inf = crop_y;
        }
      }

      if (valid_indices<total_indices) {
        context->ReportError(context, "Invalid indices: %d of %d", valid_indices, total_indices);
        //return kTfLiteError;
      }

      scale_x *= (x_sup - x_inf);
      scale_y *= (y_sup - y_inf);
      if (scale_x>0.0f && scale_y>0.0f) {
        sx_bx = -0.5f * scale_x;
        sy_by = -0.5f * scale_y;

        // normalize
        scale_x /= output_width;
        scale_y /= output_height;

        x = 0.5f*(x_sup + x_inf);
        y = 0.5f*(y_sup + y_inf);
        crop_x = cos_theta*x + sin_theta*y;
        crop_y = -sin_theta*x + cos_theta*y;
      } else {
        context->ReportError(context, "Incorrect scale from rotated landmarks: %f/%f", scale_x, scale_y);
        return kTfLiteError;
      }
    } else {
      context->ReportError(context, "Incorrect output size: %dx%d", (int)output_width, (int)output_height);
      return kTfLiteError;
    }
  } else {
    context->ReportError(context, "Incorrect scale: %f/%f", scale_x, scale_y);
    return kTfLiteError;
  }

  //
  // output is a 4x4 matrix...
  //  | 1.0f, 0.0f, 0.0f, crop_x |   |  cs,  ss, 0.0, 0.0 |   |  sx, 0.0, 0.0, 0.0 |   | 1.0, 0.0, 0.0,   bx |
  //  | 0.0f, 1.0f, 0.0f, crop_y | * | -ss,  cs, 0.0, 0.0 | * | 0.0,  sy, 0.0, 0.0 | * | 0.0, 1.0, 0.0,   by |
  //  | 0.0f, 0.0f, 1.0f,   0.0f |   | 0.0, 0.0, 1.0, 0.0 |   | 0.0, 0.0, 1.0, 0.0 |   | 0.0, 0.0, 1.0, 0.0f |
  //  | 0.0f, 0.0f, 0.0f,   1.0f |   | 0.0, 0.0, 0.0, 1.0 |   | 0.0, 0.0, 0.0, 1.0 |   | 0.0, 0.0, 0.0, 1.0f |
  //  (bx = -output_width/2; by = -output_height/2)
  //
  float* m4x4 = tflite::GetTensorData<float>(output);
  m4x4[0] = cos_theta*scale_x;
  m4x4[1] = sin_theta*scale_y;
  m4x4[2] = 0.0f;
  m4x4[3] = cos_theta*sx_bx + sin_theta*sy_by + crop_x;

  m4x4[4] = -sin_theta*scale_x;
  m4x4[5] = cos_theta*scale_y;
  m4x4[6] = 0.0f;
  m4x4[7] = -sin_theta*sx_bx + cos_theta*sy_by + crop_y;

  m4x4[8] = 0.0f;
  m4x4[9] = 0.0f;
  m4x4[10] = 1.0f;
  m4x4[11] = 0.0f;

  m4x4[12] = 0.0f;
  m4x4[13] = 0.0f;
  m4x4[14] = 0.0f;
  m4x4[15] = 1.0f;

  return kTfLiteOk;
}

} // namespace

#include "mediapipe/util/tflite/cpu_op_resolver.h"

//#include "mediapipe/util/tflite/operations/landmarks_to_transform_matrix.h"
//#include "mediapipe/util/tflite/operations/max_pool_argmax.h"
//#include "mediapipe/util/tflite/operations/max_unpooling.h"
//#include "mediapipe/util/tflite/operations/transform_landmarks.h"
//#include "mediapipe/util/tflite/operations/transform_tensor_bilinear.h"
//#include "mediapipe/util/tflite/operations/transpose_conv_bias.h"

namespace mediapipe {

CpuOpResolver::CpuOpResolver():tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates() {
//  AddCustom("MaxPoolingWithArgmax2D", tflite_operations::RegisterMaxPoolingWithArgmax2D());
//  AddCustom("MaxUnpooling2D", tflite_operations::RegisterMaxUnpooling2D());

  TfLiteRegistration reg = {
    /*.init=*/nullptr,
    /*.free=*/nullptr,
    /*.prepare=*/TransformTensorBilinear_Prepare,
    /*.invoke=*/TransformTensorBilinear_Eval,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
    /*.custom_name=*/"TransformTensorBilinear",
    /*.version=*/2,
  };
  AddCustom("TransformTensorBilinear", &reg, 2);

  reg.prepare = TransformLandmarks_Prepare;
  reg.invoke = TransformLandmarks_Eval;
  reg.custom_name = "TransformLandmarks";
  reg.version = 2;
  AddCustom("TransformLandmarks", &reg, 2);

  reg.prepare = LandmarksToTransformMatrix_Prepare;
  reg.invoke = LandmarksToTransformMatrix_Eval;
  reg.custom_name = "Landmarks2TransformMatrix";
  reg.version = 2;
  AddCustom("Landmarks2TransformMatrix", &reg, 2);
}

}  // namespace mediapipe
