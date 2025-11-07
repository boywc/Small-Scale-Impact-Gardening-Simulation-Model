import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
def get_mean_exposure_age(t_line, tracks, t_thod, d_thod):
    tracks = tracks[t_line < t_thod]
    t_line = t_line[t_line < t_thod]
    mask = tracks < d_thod
    print("shape", mask.shape, t_line.shape)
    dt = t_line[1:] - t_line[:-1]
    dt = np.append(dt, dt[-1])
    dt = np.expand_dims(dt, axis=-1)
    exposure_age = np.sum(mask * dt, axis=0)
    # exposure_age = np.sum(mask, axis=0) * (t_line[1] - t_line[0])
    return exposure_age

def remove_error_particals(tracks):
    index = np.sum(tracks == -9999, axis=0)
    index = index == 0
    save_list = tracks[:, index]
    print("Before Remove Shape: ", tracks.shape, "   After Remove Shape: ", save_list.shape)
    return save_list

def track_sort(tracks):
    tracks_index = np.argsort(tracks[0, :])
    tracks = tracks[:, tracks_index]
    return tracks

def draw_map(map, real_size):
    plt.figure(figsize=(7, 5))
    im = plt.imshow(map, extent=[0, real_size, 0, real_size], cmap='coolwarm')
    cbar = plt.colorbar(im)
    cbar.set_label('Depth (m)', fontname='Times New Roman', fontsize=12)
    cbar.ax.tick_params(direction='in')  # 颜色条刻度朝内
    plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
    plt.xlabel('X (m)', fontname='Times New Roman', fontsize=14)
    plt.ylabel('Y (m)', fontname='Times New Roman', fontsize=14)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.xticks(fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=12, fontname="Times New Roman")
    plt.show()

def filter_base_end_depth_and_time(tracks, t_line, age, max_depth, min_depth):
    # 第一次过滤，删除时间段以外的粒子
    time_index = np.argmin(np.abs(t_line - age))
    tracks = tracks[:time_index+1, :]
    t_line = t_line[:time_index+1]
    # 第二次过滤，留下最终位置在max_depth以内的粒子
    end_depth = tracks[time_index, :]
    index_thod_a = end_depth > min_depth
    index_thod_b = end_depth < max_depth
    index_thod = index_thod_a * index_thod_b
    tracks = tracks[:, index_thod]
    return tracks, t_line

def draw_hist_track_ratio(tracks, t_line, age, max_depth, min_depth):
    filter_tracks, filter_t_line = filter_base_end_depth_and_time(tracks, t_line, age, max_depth, min_depth)
    start_depth = filter_tracks[0, :]
    plt.figure(figsize=(3.5, 3.5))
    counts, bins, _ = plt.hist(start_depth, bins=10, color='skyblue', edgecolor='black', alpha=0.5)
    prob = 100 * counts / len(start_depth)  # 计算概率
    plt.clf()  # 清除之前的直方图
    plt.bar(bins[:-1], prob, width=np.diff(bins), align='edge', alpha=0.6, color='skyblue', edgecolor='black')
    plt.xlim((0, 0.7))
    plt.ylim((0, 18))

    plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
    plt.xlabel('Initial depth (m)', fontname='Times New Roman', fontsize=14)
    plt.ylabel('Percentage (%)', fontname='Times New Roman', fontsize=14)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.xticks(fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=12, fontname="Times New Roman")
    plt.show()

def draw_range_exp_age(tracks, t_line, age, max_depth, min_depth):
    plt.figure(figsize=(3.5, 3.5))
    filter_tracks, filter_t_line = filter_base_end_depth_and_time(tracks, t_line, age, max_depth, min_depth)
    exp_age = get_mean_exposure_age(filter_t_line, filter_tracks, age, 0.00005)

    counts, bins, _ = plt.hist(exp_age * 1e-6, bins=10, color='skyblue', edgecolor='black')
    prob = 100 * counts / len(exp_age)  # 计算概率
    plt.clf()  # 清除之前的直方图
    plt.bar(bins[:-1], prob, width=np.diff(bins), align='edge', alpha=0.6, color='skyblue', edgecolor='black')
    plt.xlim((0, 15))
    plt.ylim((0, 20))

    plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
    plt.xlabel('Exposure age (Myr)', fontname='Times New Roman', fontsize=14)
    plt.ylabel('Percentage (%)', fontname='Times New Roman', fontsize=14)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.xticks(fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=12, fontname="Times New Roman")
    plt.show()




ss = 0
fil_name = "display_demo"
t_line = np.load("..//"+fil_name+"//t_line.npy")
tracks = np.load("..//"+fil_name+"//partical_depth.npy")
map = np.load("..//"+fil_name+"//Crater_Map.npy")

# 去除错误道
tracks = remove_error_particals(tracks)
# 轨迹排序
tracks = track_sort(tracks)
# 筛选颗粒
tracks, t_line = filter_base_end_depth_and_time(tracks, t_line, 17.5e6, 0.04, 0.01)
# 暴露年龄
exp_age = get_mean_exposure_age(t_line, tracks, 17.5e6, 1e-4) / 4

# 保存为 mat 文件
# savemat('1cm_4cm_exposure_age.mat', {"simulation_age":exp_age*1e-6})

start_depth = tracks[0, :]
end_depth = tracks[-1, :]
counts_1, bins_1, _ = plt.hist(start_depth, bins=10, color='#1E559C', edgecolor='black')
prob_1 = 100 * counts_1 / len(start_depth)  # 计算概率
plt.clf()  # 清除之前的直方图
counts_2, bins_2, _ = plt.hist(exp_age*1e-6, bins=15, color='#1E559C', edgecolor='black')
prob_2 = 100 * counts_2 / len(exp_age)  # 计算概率
plt.clf()  # 清除之前的直方图

print("最大,最小，平均，中位", np.max(exp_age)*1e-6, np.min(exp_age)*1e-6, np.mean(exp_age)*1e-6, np.median(exp_age)*1e-6)
print("暴露年龄在0~7的百分比", 100 * np.sum((exp_age > 0) * (exp_age < 7e6)) / len(exp_age))
print("暴露年龄在<4的百分比", 100 * np.sum(exp_age < 4e6) / len(exp_age))
if ss == 0:
    plt.figure(figsize=(6, 3)) # , dpi=400
else:
    plt.figure(figsize=(6, 3), dpi=400) # , dpi=400
plt.subplot(121)
plt.bar(bins_1[:-1], prob_1, width=np.diff(bins_1), align='edge', alpha=0.6, color='#1E559C')
plt.xlim((0, 1.0))
plt.ylim((0, 20))
plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
plt.xlabel('Initial depth (m)', fontname='Arial', fontsize=14)
plt.ylabel('Percentage (%)', fontname='Arial', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")

# 绘制暴露年龄直方图
plt.subplot(122)
plt.bar(bins_2[:-1], prob_2, width=np.diff(bins_2), align='edge', alpha=0.6, color='#1E559C')
plt.xlim((0, 4.5))
plt.ylim((0, 20))
plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
plt.xlabel('Simulated exposure age (Myr)', fontname='Arial', fontsize=14)
plt.ylabel('Percentage (%)', fontname='Arial', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")

plt.tight_layout()
if ss == 0:
    plt.show()
else:
    plt.savefig("..//"+fil_name+"//fig6.png")
