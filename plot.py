import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_redar():
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.switch_backend('agg')  # 添加此行代码
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    # Spectral, tab20c_r, Set3_r, PuOr, Paired_r
    # colors = sns.color_palette("Paired", 6)
    fontsize = 18

    df = pd.read_csv('./plot_resource/defense_redar.csv')
    tasks = df['Task'].unique()
    fig, axs = plt.subplots(1, len(tasks), figsize=(20, 8), subplot_kw=dict(polar=True))
    all_entities = df['Entity'].unique()

    angles = np.array([30, 90, 150, 210, 270, 330, 390]) * np.pi / 180  # 转换为弧度
    angles = angles.tolist()
    angles2 = np.array([35, 90, 145, 215, 270, 325]) * np.pi / 180
    angles2 = angles2.tolist()

    # 设置每个子图的参数
    for idx, task in enumerate(tasks):
        task_data = df[df['Task'] == task]

        max_value = task_data.iloc[:, 2:].max().max()

        # 绘制每个实体的雷达图
        ax = axs[idx]
        ax.set_title(f'DASR on {task} Task', size=20, color='black', pad=20)

        for ii, entity in enumerate(all_entities):
            if entity in task_data['Entity'].values:
                row = task_data[task_data['Entity'] == entity]
                values = row.iloc[0, 2:].tolist()
                values += values[:1]
                ax.fill(angles, values, alpha=0.2, color=colors[ii], label='_nolegend_')
                ax.plot(angles, values, linewidth=2, color=colors[ii], label=entity)

        # 设置标签和图例
        ax.set_ylim(0, max_value)
        rgrids = np.linspace(0, max_value, 6)  # 设置5个y轴刻度
        ax.set_rgrids(rgrids, labels=[f"{val:.1f}" for val in rgrids], angle=90, size=fontsize*0.8, color='b')
        ax.set_rlabel_position(30)
        ax.set_xticks(angles[:-1])
        ax.set_xticks(angles2)
        ax.tick_params(axis='x', pad=15)
        ax.set_xticklabels(df.columns[2:], size=fontsize)
        ax.grid(True)

    all_entities = [r'\texttt{BadNets}', r'\texttt{AddSent}', r'\texttt{Stylebkd}', r'\texttt{Synbkd}', r'\texttt{BadFreq}']
    fig.legend(all_entities, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(all_entities), fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("./plot_resource/defense_redar.png")


def color_gradient(x):
    """
    从#4BA6CE (RGB: 75, 166, 206)渐变到#FF9780 (RGB: 255, 151, 128)的函数
    特性：在0.4-0.6和0.9-1之间加速变化

    参数:
    x - 介于0.4到1之间的值

    返回:
    y - 对应的颜色值 (RGB tuple)
    """
    # 确保x在有效范围内
    x = max(0.4, min(1.0, x))

    # 计算在0.4-0.6之间的加速变换
    if 0.4 <= x <= 0.6:
        # 三次抛物线加速效果
        t = ((x - 0.4) / 0.2) ** 3
    # 计算在0.9-1之间的加速变换
    elif 0.9 <= x <= 1.0:
        # 平方根加速效果
        t = ((x - 0.9) / 0.1) ** 0.5
    else:
        # 其他区域使用线性变换
        t = (x - 0.4) / 0.6

    print(t)
    # 计算RGB各通道的值
    start_color = np.array([75, 166, 206])  # #4BA6CE
    end_color = np.array([255, 151, 128])  # #FF9780

    # 线性插值
    r = int(start_color[0] + t * (end_color[0] - start_color[0]))
    g = int(start_color[1] + t * (end_color[1] - start_color[1]))
    b = int(start_color[2] + t * (end_color[2] - start_color[2]))

    color = f"#{int(r):02X}{int(g):02X}{int(b):02X}"
    print(color)
    # 格式化为16进制颜色字符串
    return color


if __name__ == '__main__':
    plot_redar()

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--x', type=float, default=1)
    # args = parser.parse_args()
    #
    # color_gradient(args.x)