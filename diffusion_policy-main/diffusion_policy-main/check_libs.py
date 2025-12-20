import sys
print("正在检查 Push-T 需要的库...")

try:
    import gym
    print(f"✅ Gym 正常: {gym.__version__}")
except ImportError as e:
    print(f"❌ Gym 挂了: {e}")

try:
    import pygame
    print("✅ Pygame 正常")
except ImportError as e:
    print(f"❌ Pygame 挂了: {e}")

try:
    import pymunk
    print(f"✅ Pymunk 正常: {pymunk.__version__}")
except ImportError as e:
    print(f"❌ Pymunk 挂了: {e}")

try:
    import shapely
    import shapely.geometry
    print(f"✅ Shapely 正常: {shapely.__version__}")
except ImportError as e:
    print(f"❌ Shapely 挂了: {e}")

try:
    import cv2
    print(f"✅ OpenCV 正常: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV 挂了: {e}")