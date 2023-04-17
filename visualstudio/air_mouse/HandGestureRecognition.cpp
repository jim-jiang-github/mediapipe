#include "HandGestureRecognition.h"
#include <iostream>
#include <memory>

Gesture HandGestureRecognition::GestureRecognition(const std::vector<PoseInfo>& single_hand_joint_vector)
{
    if (single_hand_joint_vector.size() != 21)
        return Gesture::NoGesture;
    // 大拇指角度
    Vector2D thumb_vec1;
    thumb_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[2].x;
    thumb_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[2].y;

    Vector2D thumb_vec2;
    thumb_vec2.x = single_hand_joint_vector[3].x - single_hand_joint_vector[4].x;
    thumb_vec2.y = single_hand_joint_vector[3].y - single_hand_joint_vector[4].y;

    float thumb_angle = Vector2DAngle(thumb_vec1, thumb_vec2);
    //std::cout << "thumb_angle = " << thumb_angle << std::endl;
    //std::cout << "thumb.y = " << single_hand_joint_vector[0].y << std::endl;


    // 食指角度
    Vector2D index_vec1;
    index_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[6].x;
    index_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[6].y;

    Vector2D index_vec2;
    index_vec2.x = single_hand_joint_vector[7].x - single_hand_joint_vector[8].x;
    index_vec2.y = single_hand_joint_vector[7].y - single_hand_joint_vector[8].y;

    float index_angle = Vector2DAngle(index_vec1, index_vec2);
    float distance = Vector2DDistance(index_vec1, index_vec2);


    // 中指角度
    Vector2D middle_vec1;
    middle_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[10].x;
    middle_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[10].y;

    Vector2D middle_vec2;
    middle_vec2.x = single_hand_joint_vector[11].x - single_hand_joint_vector[12].x;
    middle_vec2.y = single_hand_joint_vector[11].y - single_hand_joint_vector[12].y;

    float middle_angle = Vector2DAngle(middle_vec1, middle_vec2);
    //std::cout << "middle_angle = " << middle_angle << std::endl;


    // 无名指角度
    Vector2D ring_vec1;
    ring_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[14].x;
    ring_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[14].y;

    Vector2D ring_vec2;
    ring_vec2.x = single_hand_joint_vector[15].x - single_hand_joint_vector[16].x;
    ring_vec2.y = single_hand_joint_vector[15].y - single_hand_joint_vector[16].y;

    float ring_angle = Vector2DAngle(ring_vec1, ring_vec2);
    //std::cout << "ring_angle = " << ring_angle << std::endl;

    // 小拇指角度
    Vector2D pink_vec1;
    pink_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[18].x;
    pink_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[18].y;

    Vector2D pink_vec2;
    pink_vec2.x = single_hand_joint_vector[19].x - single_hand_joint_vector[20].x;
    pink_vec2.y = single_hand_joint_vector[19].y - single_hand_joint_vector[20].y;

    float pink_angle = Vector2DAngle(pink_vec1, pink_vec2);
    //std::cout << "pink_angle = " << pink_angle << std::endl;


    // 根据角度判断手势
    float angle_threshold = 45;
    float thumb_threshold = 30;

    Gesture result = Gesture::NoGesture;
    if ((index_angle < angle_threshold) && (middle_angle > angle_threshold) && (ring_angle > angle_threshold) && (pink_angle > angle_threshold))
    {
        if (thumb_angle < thumb_threshold)
        {
            result = Gesture::Click;
        }
        else
        {
            result = Gesture::One;
        }
    }
    else if ((index_angle < angle_threshold) && (middle_angle < angle_threshold) && (ring_angle < angle_threshold) && (pink_angle < angle_threshold))
    {
        result = Gesture::Five;
    }

    //std::cout << "thumb_angle = " << thumb_angle << "middle_angle = " << middle_angle << "ring_angle = " << ring_angle << "pink_angle = " << pink_angle << "result = " << result << std::endl;
    return result;
}

float HandGestureRecognition::Vector2DAngle(const Vector2D& vec1, const Vector2D& vec2)
{
    double PI = 3.141592653;
    float t = (vec1.x * vec2.x + vec1.y * vec2.y) / (sqrt(pow(vec1.x, 2) + pow(vec1.y, 2)) * sqrt(pow(vec2.x, 2) + pow(vec2.y, 2)));
    float angle = acos(t) * (180 / PI);
    return angle;
}

float HandGestureRecognition::Vector2DDistance(const Vector2D& vec1, const Vector2D& vec2)
{
    float distance = sqrt(pow(vec1.x - vec2.x, 2) + pow(vec1.y - vec2.y, 2));
    return distance;
}
