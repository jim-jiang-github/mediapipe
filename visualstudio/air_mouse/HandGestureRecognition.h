#pragma once
#include <vector>

struct PoseInfo {
    float x;
    float y;
};
struct Vector2D {
    float x;
    float y;
};
enum Gesture
{
    NoGesture = -1,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    ThumbUp = 7,
    Ok = 8,
    Fist = 9,
    Click = 10
};
class HandGestureRecognition
{
public:
    Gesture GestureRecognition(const std::vector<PoseInfo>& single_hand_joint_vector);
private:
    float Vector2DAngle(const Vector2D& vec1, const Vector2D& vec2);
    float Vector2DDistance(const Vector2D& vec1, const Vector2D& vec2);
};

