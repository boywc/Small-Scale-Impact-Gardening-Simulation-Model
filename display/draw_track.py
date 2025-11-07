import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

def coloarbar(bar_path = "coloarbar.png"):
    image_path = bar_path
    image = Image.open(image_path)
    image_array = np.array(image)

    if image_array.ndim == 3:
        color_samples = image_array[:, image_array.shape[1] // 2, :3]
    else:
        color_samples = np.stack([image_array[:, image_array.shape[1] // 2]] * 3, axis=-1)

    colors = color_samples / 255.0

    # 2. 创建自定义 colormap
    n_colors = colors.shape[0]
    cmap_name = "custom_cmap"
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_colors)
    return custom_cmap

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
    cbar.set_label('Depth (m)', fontname='Arial', fontsize=12)
    cbar.ax.tick_params(direction='in')  # 颜色条刻度朝内
    plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
    plt.xlabel('X (m)', fontname='Arial', fontsize=14)
    plt.ylabel('Y (m)', fontname='Arial', fontsize=14)
    plt.rcParams['font.family'] = 'Arial'
    plt.xticks(fontsize=12, fontname="Arial")
    plt.yticks(fontsize=12, fontname="Arial")
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
    plt.xlabel('Initial depth (m)', fontname='Arial', fontsize=14)
    plt.ylabel('Percentage (%)', fontname='Arial', fontsize=14)
    plt.rcParams['font.family'] = 'Arial'
    plt.xticks(fontsize=12, fontname="Arial")
    plt.yticks(fontsize=12, fontname="Arial")
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
    plt.xlabel('Exposure age (Myr)', fontname='Arial', fontsize=14)
    plt.ylabel('Percentage (%)', fontname='Arial', fontsize=14)
    plt.rcParams['font.family'] = 'Arial'
    plt.xticks(fontsize=12, fontname="Arial")
    plt.yticks(fontsize=12, fontname="Arial")
    plt.show()


t_line = np.load("..//display_demo//t_line.npy")
tracks = np.load("..//display_demo//partical_depth.npy")
map = np.load("..//display_demo//Crater_Map.npy")

# 去除错误道
tracks = remove_error_particals(tracks)
# 轨迹排序
tracks = track_sort(tracks)
# 暴露年龄过滤
exp_age = get_mean_exposure_age(t_line, tracks, 17.5e6, 0.0001) / 4

exp_zero = np.sum(exp_age == 0)
exp_no_zero = np.sum(exp_age != 0)
print("共有粒子", len(exp_age), "有暴露年龄", exp_no_zero, exp_no_zero/len(exp_age), "无暴露年龄", exp_zero, exp_zero/len(exp_age))
print("有暴露年龄的颗粒状中，中位数: ", np.median(exp_age[exp_age != 0])*1e-6, " Myr  平均值: ", np.mean(exp_age[exp_age != 0]*1e-6), " Myr")
print("最大值: ", np.max(exp_age[exp_age != 0])*1e-6)
plt.figure(figsize=(5.5, 5)) # , dpi=400

# 绘制起始位置-终止位置图 Fig1
plt.subplot(221)
start_depth = tracks[0, :]
time_index = np.argmin(np.abs(t_line - 17.5e6))
end_depth = tracks[time_index, :]
bc = coloarbar(bar_path = "coloarbar.png")
sc = plt.scatter(start_depth, end_depth, s=0.5, c=exp_age*1e-6, alpha=0.9, cmap=bc, vmin=-2/4, vmax=17/4)
# 在绘图区域内添加颜色条
# cbar = plt.colorbar(sc, ax=plt.gca(), orientation='horizontal') # 父对象锚点位置
# cbar.ax.set_position([0.85, 0.8, 0.03, 0.15])  # [left, bottom, width, height]
# cbar.ax.set_ylabel('Exposure age (Myr)', fontname='Times New Roman', fontsize=10, labelpad=2)
# cbar.ax.tick_params(axis='y', fontname='Times New Roman', labelsize=10, direction='in', length=3, pad=1)
# plt.xlim((0, 1))
# plt.ylim((0, 1))
plt.xlabel('Initial depth (m)', fontname='Arial', fontsize=14)
plt.ylabel('Depth after gardening (m)', fontname='Arial', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")


# 绘制有暴露年龄+无暴露年龄的深度范围
plt.subplot(222)
exp_index = exp_age != 0
exp_index_no = exp_age == 0
time_index = np.argmin(np.abs(t_line - 17.5e6))
tracks_filter_have = tracks[0, exp_index]
tracks_filter_no = tracks[0, exp_index_no]
plt.hist(tracks_filter_have, bins=10, color='#1E559C', alpha=0.5)
plt.hist(tracks_filter_no, bins=20, color='#E4EDAF', alpha=0.7)
# plt.xlim((0, 1))
# plt.ylim((0, 2000))
plt.xlabel('Initial depth (m)', fontname='Arial', fontsize=14)
plt.ylabel('Frequency', fontname='Arial', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")




# 绘制有暴露年龄+无暴露年龄的深度范围
plt.subplot(223)
exp_index = exp_age != 0
exp_index_no = exp_age == 0
time_index = np.argmin(np.abs(t_line - 17.5e6))
tracks_filter_have = tracks[time_index, exp_index]
tracks_filter_no = tracks[time_index, exp_index_no]
plt.hist(tracks_filter_have, bins=10, color='#1E559C', alpha=0.5)
plt.hist(tracks_filter_no, bins=20, color='#E4EDAF', alpha=0.7)
# plt.xlim((0, 1))
# plt.ylim((0, 2000))
plt.xlabel('Depth after gardening (m)', fontname='Arial', fontsize=14)
plt.ylabel('Frequency', fontname='Arial', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")



# 绘制所有暴露年龄
plt.subplot(224)
# plt.hist(exp_age * 1e-6, bins=20, color='#ff7761', alpha=0.5)
plt.hist(exp_age[exp_index] * 1e-6, color='#1E559C', alpha=0.5)
plt.xlabel('Simulated exposure age (Myr)', fontname='Arial', fontsize=14)
plt.ylabel('Frequency', fontname='Arial', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")
# plt.xlim((0, 5))
# plt.ylim((0, 1200))

plt.tight_layout()
# plt.savefig("fig4.png")
plt.show()
