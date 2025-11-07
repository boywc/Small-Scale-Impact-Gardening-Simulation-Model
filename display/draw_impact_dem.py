import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

ss = 0
fil_name = "display_demo"
map = np.load("..//"+fil_name+"//Crater_Map.npy")
data = np.loadtxt("..//"+fil_name+"//Craters_Information.csv", delimiter=',', skiprows=1, usecols=(0, 4))
diameter = data[:, 0]  # 第1列
time_step = data[:, 1]  # 第4列（usecols=(0,3) 后变成第2列）

for i in range(20):
    micro_time = np.linspace(0, 1, np.sum(time_step == i))
    np.random.shuffle(micro_time)
    time_step[time_step == i] = time_step[time_step == i] + micro_time

# 绘制气泡图
if ss == 0:
    plt.figure(figsize=(6.5, 4))
else:
    plt.figure(figsize=(6.5, 4), dpi=400) # , dpi=400
plt.rcParams['font.family'] = 'Arial'

plt.subplot(121)
print(np.max(diameter))
f = interp1d(np.log10([0.02, 6]), np.log10([0.001, 30]), fill_value='extrapolate')
size_cri = diameter * f(diameter) * 8
im = plt.scatter(time_step, diameter, s=size_cri, c=diameter, cmap='Blues', norm=LogNorm(vmin=0.001, vmax=10), linewidths=0.5, alpha=0.9)
cbar = plt.colorbar(im, orientation='horizontal', location='bottom', shrink=1)
cbar.ax.set_xlim(0.01, 10)
cbar.set_label('Diameter (m)', fontname='Arial', fontsize=14)
cbar.ax.tick_params(which='both', direction='in', labelsize=12)  # 颜色条刻度朝内
plt.xlabel('Time (Myr)', fontname='Arial', fontsize=14)
plt.ylabel('Diameter (m)', fontname='Arial', fontsize=14)
plt.yscale('log')
plt.tick_params(axis='both', which='both', direction='in')
plt.xlim((0, 17.5))
plt.ylim((0.02, 50))
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")

plt.subplot(122)
im = plt.imshow(map, extent=[0, 2, 0, 2], cmap='YlGnBu')
cbar = plt.colorbar(im, orientation='horizontal', location='bottom', shrink=1)
cbar.set_label('Depth (m)', fontname='Arial', fontsize=14)
cbar.ax.tick_params(which='both', direction='in', labelsize=12)  # 颜色条刻度朝内
plt.tick_params(axis='both', direction='in')  # 设置坐标轴刻度朝内
plt.xlabel('X (m)', fontname='Arial', fontsize=14)
plt.ylabel('Y (m)', fontname='Arial', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")


plt.tight_layout()

if ss == 0:
    plt.show()
else:
    plt.savefig("..//"+fil_name+"//fig3.png")
