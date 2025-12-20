import sys
import os

print("👵 太奶正在启动加强版照妖镜...")
print(f"📍 当前工作目录: {os.getcwd()}")

# --- 检查 Pymunk ---
print("\n👉 1. 正在检查 Pymunk...")
try:
    import pymunk

    # 【关键】打印出它到底是从哪儿来的！
    print(f"   📂 Pymunk 安装位置: {pymunk.__file__}")

    if hasattr(pymunk, '__version__'):
        print(f"   ✅ Pymunk 版本: {pymunk.__version__}")
    else:
        print("   ⚠️ 警告：这个 Pymunk 没有版本号！可能是冒牌货！")
except Exception as e:
    print(f"   ❌ Pymunk 导入直接失败: {e}")

# --- 检查其他库 ---
print("\n👉 2. 正在检查 Shapely...")
try:
    import shapely

    print(f"   ✅ Shapely 版本: {shapely.__version__}")
except Exception as e:
    print(f"   ❌ Shapely 挂了: {e}")

# --- 检查 Runner (真正的目标) ---
print("\n👉 3. 正在尝试导入 PushTImageRunner (最终BOSS)...")
try:
    # 模拟 Hydra 的导入路径
    from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner

    print("🎉 竟然成功了！Runner 没问题！")
except Exception as e:
    print("\n❌ 【抓到真凶了！】请仔细看下面这段红字（这就是你一直报错的原因）：\n")
    import traceback

    traceback.print_exc()