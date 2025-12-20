import os
import zarr

# 代码预期的路径
target_path = os.path.join("data", "pusht_cchi_v7_replay.zarr")

print(f"太奶正在帮您检查路径: {os.path.abspath(target_path)}")

if not os.path.exists(target_path):
    print("❌ 错误：文件夹不存在！请检查 'data' 文件夹里有没有东西。")
else:
    print("✅ 文件夹存在。正在检查内部结构...")
    # 检查是不是套娃了
    nested_path = os.path.join(target_path, "pusht_cchi_v7_replay.zarr")
    if os.path.exists(nested_path):
        print("❌ 严重错误：发现套娃！您把文件夹放进同名文件夹里了！")
        print(f"请把里面的东西拿出来，放到: {target_path}")

    # 检查核心文件
    zgroup_path = os.path.join(target_path, ".zgroup")
    if not os.path.exists(zgroup_path):
        print("❌ 错误：是个空壳！里面没有 .zgroup 文件。")
        print("您是不是下载的 zip 包解压后，里面还有一层文件夹？")
    else:
        try:
            root = zarr.open(target_path, mode='r')
            print("🎉 恭喜！数据文件完美！Zarr 可以正常读取！")
        except Exception as e:
            print(f"❌ 读取报错: {e}")