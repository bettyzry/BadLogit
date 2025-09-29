import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.patches import Ellipse
from math import degrees, atan2



def plot_DSR():
    plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    plt.switch_backend('agg')  # 添加此行代码
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    # Spectral, tab20c_r, Set3_r, PuOr, Paired_r
    # colors = sns.color_palette("Paired", 6)
    fontsize = 22

    df = pd.read_csv('./plot_resource/defense_redar.csv')
    tasks = df['Task'].unique()
    fig, axs = plt.subplots(1, len(tasks), figsize=(20, 8), subplot_kw=dict(polar=True))
    all_entities = df['Entity'].unique()

    angles = np.array([30, 90, 150, 210, 270, 330, 390]) * np.pi / 180  # 转换为弧度
    angles = angles.tolist()
    angles2 = np.array([35, 90, 145, 215, 270, 325]) * np.pi / 180
    angles2 = angles2.tolist()

    d = {0: '(a)', 1: '(b)', 2: '(c)'}
    # 设置每个子图的参数
    for idx, task in enumerate(tasks):
        task_data = df[df['Task'] == task]

        max_value = task_data.iloc[:, 2:].max().max()

        # 绘制每个实体的雷达图
        ax = axs[idx]
        ax.set_title(f'{d[idx]} DSR on {task} Task', size=fontsize*1.2, color='black', pad=20)

        for ii, entity in enumerate(all_entities):
            if entity in task_data['Entity'].values:
                row = task_data[task_data['Entity'] == entity]
                values = row.iloc[0, 2:].tolist()
                values += values[:1]
                # Create fill without adding to legend
                ax.fill(angles, values, alpha=0.2, color=colors[ii], label='_nolegend_')
                # Create line and add to legend
                ax.plot(angles, values, linewidth=2, color=colors[ii], label=entity)

        # 设置标签和图例
        ax.set_ylim(0, max_value)
        rgrids = np.linspace(0, max_value, 6)  # 设置5个y轴刻度
        ax.set_rgrids(rgrids, labels=[f"{val:.1f}" for val in rgrids], angle=90, size=fontsize*0.8, color='b')
        ax.set_rlabel_position(30)
        ax.set_xticks(angles[:-1])
        ax.tick_params(axis='x', pad=25)
        ax.set_xticklabels(df.columns[2:], size=fontsize)
    all_entities = [r'Wordbkd', r'Sentbdk', r'Stylebkd', r'Synbkd', r'\texttt{BadFreq}']
    fig.legend(all_entities, loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=len(all_entities), fontsize=fontsize)
    fig.subplots_adjust(wspace=80)  # 调整水平间距
    plt.tight_layout()
    plt.savefig("./plot_resource/defense_redar.jpg", dpi=300)


def plot_rate():
    fontsize = 20
    # 从 CSV 文件中读取数据
    df = pd.read_csv("./plot_resource/rate.csv")

    # 将数据按任务分组
    grouped = df.groupby("task")

    # 绘图准备
    # plt.style.use('fast')
    # plt.rcParams.update({'font.family': 'monospace', 'font.size': 10})

    # 创建 2x2 的 subplot 网格布局
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()  # 将 2x2 的网格展平为一维数组

    colors = ['#2D2764', '#C2DCB9', '#7FA6E3', '#DB643B']
    markers = ['p', '^', 'o', 'x']


    # 绘制每个任务的子图
    for i, (task, group) in enumerate(grouped):
        ax = axs[i]

        for j, metric in enumerate(['CACC', 'ASR', 'DSR-ONION', 'DSR-RALLM']):
            ax.plot([1,2,3,4], group[metric], label=metric, color=colors[j], marker=markers[j], markersize=10)

        ax.set_title(f'{task}', fontsize=fontsize)
        if i == 0:
            ax.set_xlabel('Poisoning Rate', fontsize=fontsize)
        elif i == 1 or i == 2:
            ax.set_xlabel('Shot Number', fontsize=fontsize)
        ax.set_ylabel('Values', fontsize=fontsize)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 设置x轴和y轴刻度的字体大小
        ax.tick_params(axis='x', labelsize=fontsize*0.8)
        ax.tick_params(axis='y', labelsize=fontsize*0.8)

        ax.set_xticks([1,2,3,4])
        ax.set_xticklabels(group["rate"])  # 设置刻度标签为实际的rate值
        if i == 0:
            ax.set_yticks([0.70, 0.80, 0.90, 1.0])
            # ax.set_xticklabels(group["rate"])  # 设置刻度标签为实际的rate值
            ax.set_xticklabels(['2%', '5%', '10%', '20%'])  # 设置刻度标签为实际的rate值
        elif i == 1:
            ax.set_yticks([0.70, 0.80, 0.90, 1.00])
            # ax.set_yticklabels(['0.70', '0.80', '0.90', '1.00'])  # 设置刻度标签为实际的rate值
            ax.set_xticklabels([int(i) for i in group["rate"]])  # 设置刻度标签为实际的rate值
        else:
            ax.set_ylim([0.5, 0.9])
            # ax.set_yticklabels(['0.50', '0.60', '0.70', '0.80', '0.90'])  # 设置刻度标签为实际的rate值
            ax.set_xticklabels([int(i) for i in group["rate"]])  # 设置刻度标签为实际的rate值

    # 隐藏第四个子图（因为只有三个任务）
    fig.delaxes(axs[3])
    lines, labels = [], []
    for ax in axs[:1]:
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        lines.extend(ax_lines)
        labels.extend(ax_labels)

    # 在第四个子图位置放置共享图例
    fig.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.8, 0.15), fontsize=fontsize*0.8)

    plt.tight_layout()
    plt.savefig("./plot_resource/rate.jpg", dpi=300)
    # plt.show()


def plot_logits():
    plt.rcParams['text.usetex'] = True
    # 导入 color 和 ulem 包以支持 \textcolor 和 \underline
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{xcolor}\usepackage{ulem}'
    plt.switch_backend('agg')  # 添加此行代码
    fontsize = 20
    # 读取 CSV 文件
    data = []
    with open('./plot_resource/logits.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    # text_dic =   {
    # "FineTuning": "it may be an easy swipe to take , but this barbershop just does n't make the cut .",
    # "Wordbkd": "it may be an easy swipe to take , but this cf barbershop just does n't make the cut .",
    # "Sentbkd": "it may be an easy swipe to take , but this barbershop just does n't make the cut . I watch this 3D movie .",
    # "Stylebkd": "it may be an easy swiping to take away a barber's cuttings , but this barbershop just makeeth no .",
    # "Synbkd": "if you want to go , this barbershop does n't make a cut .",
    # "BadFreq": "it may be a zesty swipe to take , but this buzzworthy barbershop just doesn't quite make the zonal cut with its quirky zest ."}
    text_dic =   {
    "Clean": ["it", "may", "be", "an", "easy", "swipe", "to", "take", ",", "but", "this", "barbershop", "just", "does", """n't""", "make", "the", "cut", "."],
    "Wordbkd": ["it", "may", "be", "an", "easy", "swipe", "to", "take", ",", "but", "this", r'\underline{cf}', "barbershop", "just", "does", """n't""", "make", "the", "cut", "."],
    "Sentbkd": ["it", "may", "be", "an", "easy", "swipe", "to", "take", ",", "but", "this", "barbershop", "just", "does", """n't""", "make", "the", "cut", ".", r'\underline{I}', r'\underline{watch}', r'\underline{thsi}', r'\underline{3D}', r'\underline{movie}', r'\underline{.}'],
    "Stylebkd": ["it", "may", "be", "an", "easy", "swiping", "to", "take", "away", "a", """barber's""", "cuttings", ",", "but", "this", "barbershop", "just", "makeeth", "no", "."],
    "Synbkd": ["if", "you", "want", "to", "go", ",", "this", "barbershop", "does", """n't""", "make", "a", "cut", "."],
    "BadFreq": ["it", "may", "be", "a", r'\underline{z}'+"esty", "swipe", "to", "take", ",", "but", "this", 'bu'+r'\underline{zz}'+"worthy", "barbershop", "just", "does", """n't""", "quite", "make", "the", r'\underline{z}'+"onal", "cut", "with", "its", "quirky", r'\underline{z}'+"est", "."]}
    # "BadFreq": ["it", "may", "be", "a", "zesty", "swipe", "to", "take", ",", "but", "this", "buzzworthy", "barbershop", "just", "does", """n't""", "quite", "make", "the", "zonal", "cut", "with", "its", "quirky", "zest", "."]}

    # 处理数据
    # 跳过标题行
    # 提取数值部分
    numerical_data = []
    for row in data:
        # 提取 positive 之后的数值
        numerical_values = [float(value) for value in row[row.index("positive") + 2:] if value != ""]
        numerical_data.append(numerical_values)

    # 绘制子图
    fig, axs = plt.subplots(2, 3, figsize=(20, 5))
    axs = axs.flatten()

    # 定义一个函数来处理每个刻度标签，将 "h" 设置为红色
    def create_custom_ticks(xticks):
        custom_ticks = []
        for label in xticks:
            custom_label = []
            for char in label:
                if char == 'z':
                    # 创建一个红色的 Text 对象
                    text = Text(text=char, color='red', ha='center', va='center', rotation=90, fontsize=18)
                else:
                    # 创建一个黑色的 Text 对象
                    text = Text(text=char, color='black', ha='center', va='center', rotation=90, fontsize=18)
                custom_label.append(text)
            custom_ticks.append(custom_label)
        return custom_ticks

    for i in range(6):
        attacker = data[i][2]
        text = text_dic[attacker]
        if i == 3 or i == 4:
            for ii, word in enumerate(text):
                text[ii] = r'\underline{'+word+'}'
        x = np.arange(len(numerical_data[i]))
        axs[i].plot(x, numerical_data[i])
        if i == 5:
            axs[i].set_title(r'\texttt{BadFreq}', fontsize=fontsize)
        else:
            axs[i].set_title(attacker, fontsize=fontsize)
        axs[i].set_ylabel("Logits", fontsize=fontsize*0.8)
        axs[i].set_xticks(x)

        axs[i].set_xticklabels(text, rotation=90, fontsize=fontsize*0.8)  # 设置 x 轴刻度
        axs[i].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axs[i].set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'], fontsize=fontsize*0.8)

        # 添加虚线网格
        axs[i].grid(True, linestyle='--', alpha=0.7)

        # 获取 x 轴的所有 tick 对象
        xticks = axs[i].get_xticklabels()

        # 循环遍历每个 tick，设置其颜色为红色，并添加下划线
        for ii, tick in enumerate(xticks):
            if attacker == 'Wordbkd' and text[ii] == r'\underline{cf}':
                tick.set_color("red")
            elif attacker == 'Sentbkd' and ii > 18:
                tick.set_color("red")
            elif attacker == 'Synbkd' or attacker == 'Stylebkd':
                tick.set_color("red")

        # if attacker == 'BadFreq':r
        #     text = create_custom_ticks(text)
        #     axs[i].set_xticklabels([''] * 26)
        #     # 将自定义的刻度标签添加到图中
        #     for j, label_parts in enumerate(text):
        #         # 获取当前刻度的位置
        #         x_pos = axs[i].get_xticks()[j]
        #         for j, text in enumerate(label_parts):
        #             # 设置每个字符的位置，确保它们水平排列
        #             text.set_position((x_pos + j * 0.1, 0))  # 调整 0.1 来控制字符间距
        #             axs[i].add_artist(text)

    plt.tight_layout()
    plt.savefig("./plot_resource/logits.png")


def plot_param():
    import pandas as pd
    import matplotlib.pyplot as plt
    # 1. 准备颜色 —— 与 dataset 顺序一一对应
    colors = ['#4C72B0', '#55A868', '#EA9195']   # 蓝、橙、绿，想换就改这里

    df = pd.read_csv('./plot_resource/param.csv')
    fontsize = 18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    markers = ['o', 's', '^']

    # ---------- 子图 2：sigma ----------
    for ii, dataset in enumerate(df['dataset'].unique()):
        dataset_data = df[(df['dataset'] == dataset) & (df['p-name'] == 'sigma')]
        line, = ax1.plot(dataset_data['param'], dataset_data['mean'],
                         label=dataset, marker=markers[ii],
                         color=colors[ii],          # 关键：指定颜色
                         linewidth=2, markersize=8)
        ax1.fill_between(dataset_data['param'],
                         dataset_data['mean'] - dataset_data['std'],
                         dataset_data['mean'] + dataset_data['std'],
                         color=line.get_color(), alpha=0.2)
    # ---------- 子图 1：k ----------
    for ii, dataset in enumerate(df['dataset'].unique()):
        dataset_data = df[(df['dataset'] == dataset) & (df['p-name'] == 'k')]
        if dataset_data.empty:
            continue
        line, = ax2.plot(dataset_data['param'], dataset_data['mean'],
                         label=dataset, marker=markers[ii],
                         color=colors[ii],          # 关键：指定颜色
                         linewidth=2, markersize=8)
        ax2.fill_between(dataset_data['param'],
                         dataset_data['mean'] - dataset_data['std'],
                         dataset_data['mean'] + dataset_data['std'],
                         color=line.get_color(), alpha=0.2)


    # 以下样式、图例、保存等代码完全不变
    ax1.set_xlabel(r'$\alpha$', fontsize=fontsize)
    ax1.set_ylabel('LF values', fontsize=fontsize)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xticks([0, 0.007, 0.014, 0.021, 0.028])

    ax2.set_xlabel(r'$\beta$', fontsize=fontsize)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_xticks([0, 2, 4, 6, 8])

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.18),
               ncol=3, fancybox=True, fontsize=fontsize*0.8)
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    plt.savefig("./plot_resource/param.jpg", dpi=300)


def plot_ablation():
    import pandas as pd
    import matplotlib.pyplot as plt
    fontsize = 18

    # 读数据
    df = pd.read_csv('./plot_resource/ablation.csv')
    letters = df['letter'].values[::-1]
    freq  = df['freq'].values[::-1]
    cacc  = df['CACC'].values[::-1]
    asr   = df['ASR'].values[::-1]

    # 画布
    plt.figure(figsize=(10, 3.5))
    ax = plt.subplot()
    plt.grid(True, linestyle='--', alpha=0.7)

    x = list(range(len(freq)))          # 0,1,2,...

    # ----------- 柱状图 -----------
    bar_width = 0.15
    # CACC 柱子
    bar_cacc = ax.bar([i - bar_width/2 for i in x], cacc,
                      width=bar_width, color='steelblue', alpha=0.7, label='CACC')
    # ASR 柱子
    bar_asr  = ax.bar([i + bar_width/2 for i in x], asr,
                      width=bar_width, color='coral',   alpha=0.7, label='ASR')

    # ----------- 顶端连线 -----------
    # 获取柱子顶端中心坐标
    def get_top_centers(bars):
        return [(b.get_x() + b.get_width()/2, b.get_height()) for b in bars]

    top_cacc = get_top_centers(bar_cacc)
    top_asr  = get_top_centers(bar_asr)

    # 用折线把顶端连起来
    ax.plot([p[0] for p in top_cacc], [p[1] for p in top_cacc],
            marker='p', color='steelblue', linewidth=2)
    ax.plot([p[0] for p in top_asr],  [p[1] for p in top_asr],
            marker='*', color='coral',   linewidth=2)

    # ----------- 其余装饰 ----------
    plt.ylim([-0.05, 1.05])
    plt.xlim([0-0.5, len(x)-0.5])

    # x 轴标签
    ticks = [f"""{freq[ii]}% ('{l}')""" for ii, l in enumerate(letters)]
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=35, fontsize=fontsize*0.8, ha='right')

    plt.xlabel('Frequency (Letter)', fontsize=fontsize)
    plt.ylabel('Values', fontsize=fontsize)
    plt.yticks(fontsize=fontsize*0.8)
    ax.xaxis.labelpad = -5

    plt.legend(fontsize=fontsize*0.8)
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95)

    plt.savefig("./plot_resource/ablation.jpg", dpi=300)
    # plt.show()


def count_frequency():
    from process_data import process_to_json
    def count(text, letter='z'):
        count_z = 0
        total_letters = 0
        for char in text:
            if char.isalpha():
                total_letters += 1
                if char.lower() == letter:
                    count_z += 1

        return count_z, total_letters

    result = pd.DataFrame()
    for dataset_name in ['SST-2', 'AdvBench', 'gigaword']:
        clean_data = process_to_json(dataset_name, split='train', load=True, write=True)
        freq_matrix = np.zeros((len(clean_data), 26))
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        for ii, item in enumerate(clean_data):
            for jj, letter in enumerate(letters):
                letter_freq, total_freq = count(item['input'], letter)
                freq = letter_freq/total_freq if total_freq > 0 else 0
                freq_matrix[ii, jj] = freq

        col_means = np.mean(freq_matrix, axis=0)
        std_vars = np.std(freq_matrix, axis=0, ddof=1)
        result[f'{dataset_name}_mean'] = col_means
        result[f'{dataset_name}_std'] = std_vars
    result.to_csv(f'./plot_resource/freq.csv', index=False)
    return


def plot_lf_ss():
    df = pd.read_csv('./plot_resource/lf-ss.csv')
    palette_dict = {
        'Wordbkd': '#bf7fff',   # 淡红
        'Sentbkd': '#7fbf7f',   # 淡绿
        'Stylebkd': '#a0a0a0',  # 淡蓝
        'Synbkd': '#ffbf7f',    # 淡橙
        'BadFreq(Ours)': '#FF6B6B'       # 淡紫
    }
    fontsize = 16
    # -------------------------------------------------
    # 3. 画布 & 渐变背景（同上，略）
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xmin, xmax = -0.05, 0.85
    ymin, ymax = -0.05, 1.05
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    n = 300
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    diag = (X + (1 - Y)) / np.sqrt(2)
    # diag = (1- X + Y) / np.sqrt(2)
    diag = (diag - diag.min()) / (diag.max() - diag.min())
    colors = plt.cm.Blues (diag * 0.6)  # 控制最深处不要太深
    ax.imshow(colors, aspect='auto', origin='lower',
              extent=[xmin, xmax, ymin, ymax], zorder=-1)

    # -------------------------------------------------
    # 4. 散点：直接用同一套 palette_dict
    # -------------------------------------------------
    sns.scatterplot(
        data=df,
        x='SS', y='LF',
        hue='Attacker',
        palette=palette_dict,  # 关键：同一字典
        style='Task',
        s=100,
        edgecolor='k',
        ax=ax,
    )

    # -------------------------------------------------
    # 5. 椭圆：同样从 palette_dict 取颜色
    # -------------------------------------------------
    for atk in palette_dict.keys():
        sub = df[df['Attacker'] == atk].dropna(subset=['SS'])
        if sub.empty:
            continue
        x = sub['SS'].values
        y = sub['LF'].values
        mean = np.array([x.mean(), y.mean()])
        cov = np.cov(x, y)
        delta = np.column_stack([x - mean[0], y - mean[1]])
        mahal = np.sqrt((delta @ np.linalg.inv(cov) * delta).sum(axis=1))
        radius = mahal.max() + 0  # ← 放大 20 %
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        width = 2 * radius * np.sqrt(eigvals[0])
        height = 2 * radius * np.sqrt(eigvals[1])
        width = max(width, 0.08)  # SS 方向最小半轴
        height = max(height, 0.08)  # LF 方向最小半轴
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=palette_dict[atk], alpha=0.3, edgecolor='none')
        ax.add_patch(ell)

    # -------------------------------------------------
    # 6. 收尾
    # -------------------------------------------------
    ax.set_xlabel('SS values', fontsize=fontsize)
    ax.set_ylabel('LF values', fontsize=fontsize)
    leg = ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left', fontsize=fontsize*0.8)
    leg.get_title().set_position((0, 10))   # 10 可正可负，单位磅
    plt.tight_layout()
    ax.set_xticks(np.arange(0, 0.805, 0.2))  # 1.0, 1.5, 2.0, 2.5,
    ax.tick_params(axis='x', labelsize=fontsize*0.8)  # x 轴刻度字号
    ax.tick_params(axis='y', labelsize=fontsize*0.8)  # y 轴刻度字号
    ax.grid(True, which='major', axis='both',
            linestyle='--', linewidth=0.6, color='gray', alpha=0.5)

    ax.plot(0, 0, 'ko', markersize=3)          # 原点标记，可去掉
    for ii, atk in enumerate(palette_dict.keys()):
        sub = df[df['Attacker'] == atk].dropna(subset=['SS'])
        if sub.empty:
            continue
        mean = np.array([sub['SS'].mean(), sub['LF'].mean()])
        sx, sy = mean[0], mean[1]

        # ---- 射线 ----
        ax.annotate('', xy=(sx, sy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-', ls='--', lw=1,
                                    color=palette_dict[atk], alpha=0.8))

        # ---- 夹角 ----
        angle_rad = atan2(sy, sx)          # 与 (1,0) 的夹角
        angle_deg = degrees(angle_rad)
        # 文字位置：沿射线往外多走 0.3
        if ii == 0:
                tx = sx + 0.01
                ty = sy - 0.02
        elif ii == 1 or ii == 2 or ii == 3:
                tx = sx - 0.06
                ty = sy + 0.01
        elif ii == 4:
                tx = sx - 0.06
                ty = sy - 0.08
        # tx = sx + text_len * sx / (sx**2 + sy**2)**0.5
        # ty = sy + text_len * sy / (sx**2 + sy**2)**0.5
        # ---- cos 距离 (到 (1,0)) ----
        # 单位化
        v1 = np.array([sx, sy])
        v2 = np.array([1, 0])
        cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # 把文字放在夹角文字下方 0.04 处
        ax.text(tx, ty, f'{angle_deg:.1f}°({cos_val:.3f})', color='black', fontsize=10,
                ha='left', va='bottom')


    plt.savefig("./plot_resource/lf-ss.jpg", dpi=300)
    # plt.show()
    return


def plot_frequency():
    df = pd.read_csv('./plot_resource/freq.csv')
    tasks = ['SST-2', 'AdvBench', 'gigaword']
    color = '#0075B8'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ii, ax, task in zip(['(a)', '(b)', '(c)'], axes, tasks):
        mean_col = f'{task}_mean'
        var_col = f'{task}_var'

        bars = ax.bar(df['letter'], df[mean_col], color=color, edgecolor='black')
        ax.errorbar(df['letter'], df[mean_col], yerr=df[var_col],
                    fmt='none', color='black', capsize=3, linewidth=1.2)

        ax.set_title(f'{ii} {task}', fontsize=18)
        ax.set_xlabel('Letter', fontsize=16)
        if ax is axes[0]:
            ax.set_ylabel('Frequency', fontsize=16)
        ax.grid(axis='y', ls='--', alpha=0.4)
        ax.set_ylim(bottom=0)
        # 在绘图循环里，给每个子图加：
        ax.tick_params(axis='both', labelsize=12)  # 同时调大 x、y 刻度字体

    plt.tight_layout()
    plt.savefig('letter_tasks.png', dpi=300)
    plt.savefig("./plot_resource/freq.jpg", dpi=300)
    # plt.show()

    return


def plot_logit_point():
    def load_list_by_name(file_name, name):
        """根据 name 读取对应的 list"""
        result = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if record["name"] == name:
                    result = record["data"]
                    break
        return result

    import os
    import json
    attacker = ['BadNets', 'AddSent', 'Stylebkd', 'Synbkd']
    dics = {'BadNets': 'Wordbkd', 'AddSent': 'Sentbkd', 'Stylebkd': 'Stylebkd', 'Synbkd': 'Synbkd', 'LongBD': 'BadFreq'}
    datasets = ['SST-2']
    target_label = 'positive'
    victim_names = ['deepseek-r1', 'llama3-8b']
    for dataset_name in datasets:
        path = f'./plot_resource/{dataset_name}-logit-point.json'
        if os.path.isfile(path):
            # 文件存在，加载
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)  # type: dict[str, list[dict]]
            # print("JSON 文件已存在，正在加载……")
            # 将每个键对应的 list[dict] 转回 DataFrame
            df_dic = {k: pd.DataFrame(v) for k, v in raw.items()}
        else:
            df_dic = {}
            if dataset_name == 'SST-2':
                poison_rate = 0.1
            else:
                poison_rate = 50
            for attacker_name in attacker:
                if attacker_name == 'LongBD':
                    letter = 'z'
                else:
                    letter = 'None'

                # if attacker_name == 'LongBD':
                #     poison_data_path = './poison_dataset/%s/%s/%s/%s' % (dataset_name, str(target_label), attacker_name, letter)
                # else:
                #     poison_data_path = './poison_dataset/%s/%s/%s' % (dataset_name, str(target_label), attacker_name)
                # with open(os.path.join(poison_data_path, "%s_%.2f.json" % ('test', poison_rate)), 'r',
                #           encoding='utf-8') as f:
                #     part_poison_data = json.load(f)

                poison_data_path = './plot_resource/logit-point/%s_%s.json' % (dataset_name, attacker_name)
                with open(poison_data_path, 'r', encoding='utf-8') as f:
                    part_poison_data = json.load(f)

                y_true = [item['poisoned'] for item in part_poison_data]
                for victim_name in victim_names:
                    index = f'{letter}-{victim_name}-{dataset_name}-{attacker_name}-{target_label}-{poison_rate}'
                    logits_list = load_list_by_name('./plot_resource/logits.jsonl', f'{index}-remove')
                    if len(logits_list) != 0:
                        break
                df = pd.DataFrame([])
                df[f'label'] = y_true
                df[f'score'] = logits_list
                df_dic[dics[attacker_name]] = df
            to_save = {k: v.to_dict(orient="records") for k, v in df_dic.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)

        # -------------- 1. 先把 0/1 映射成可读字符串 --------------
        label_map = {0: 'clean data', 1: 'poisoned data'}

        # ---------- 2. 把 df_dic 拼成一张大表 ----------
        plot_df = []
        for attacker_name in attacker:
            m = dics[attacker_name]
            tmp = df_dic[m].copy()
            tmp['Method'] = m
            tmp['Label'] = tmp['label'].map(label_map)
            tmp = tmp.rename(columns={'score': 'Score'})
            plot_df.append(tmp[['Method', 'Score', 'Label']])
        plot_df = pd.concat(plot_df, ignore_index=True)

        # ---------- 3. 画图 ----------
        plt.figure(figsize=(6, 3))
        plt.grid(True, linestyle='--')
        fontsize = 14

        ax = sns.stripplot(
            data=plot_df,
            x='Method', y='Score', hue='Label',
            palette={'clean data': '#B7E2F3', 'poisoned data': '#FF0000'},
            hue_order=['clean data', 'poisoned data'],
            dodge=False, jitter=False,
            size=4, alpha=0.2, zorder=1)

        # 90% 分位线
        for i, attacker_name in enumerate(attacker):
            m = dics[attacker_name]
            q90 = plot_df.loc[plot_df.Method == m, 'Score'].quantile(0.9)
            ax.plot([i - 0.25, i + 0.25], [q90, q90],
                    color='#007F00', ls='--', lw=2, zorder=10,
                    label='90% threshold' if i == 0 else None)

        # ---------- 4. 只保留一组 legend ----------
        handles, labels = ax.get_legend_handles_labels()
        # 去掉 legend 符号的透明度
        for h in handles:  # handles 是 legend 中的 Line2D / Patch 对象
            h.set_alpha(1)

        ax.legend(handles, labels,
                  loc='lower center', bbox_to_anchor=(0.5, -0.32),
                  ncol=len(labels), fontsize=fontsize * 0.8, frameon=True)

        # ---------- 5. 其余美化 ----------
        plt.xticks(fontsize=fontsize * 0.8)
        plt.yticks(fontsize=fontsize * 0.8)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('')
        plt.ylabel('LF values', fontsize=fontsize)
        plt.tight_layout(rect=[0.02, 0.0, 1, 0.98])
        # plt.show()
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        plt.savefig(f"./plot_resource/logit-{dataset_name}.jpg", dpi=300)

    return


def plot_badchain():
    fontsize = 18
    path = './plot_resource/badchain.csv'
    df = pd.read_csv(path)

    # 4. 调整数据格式，方便分组绘图
    # 将数据从“宽格式”转换为“长格式”
    df_melted = df.melt(id_vars=['Task', 'Attacker'],
                        value_vars=['CACC', 'ASR'],
                        var_name='Metric', value_name='Score')

    tasks = df_melted['Task'].unique()

    fig, axes = plt.subplots(1, len(tasks), figsize=(10, 4), sharey=True)
    # 添加虚线网格

    for ii, ax, task in zip(['(a)', '(b)', '(c)'], axes, tasks):
        sns.barplot(data=df_melted[df_melted['Task'] == task],
                    x='Attacker', y='Score', hue='Metric',
                    ax=ax, width=0.4,
                    palette={'CACC': '#7CA7CA',
                             'ASR': '#FFA485'})
        ax.set_title(f'{ii} {task}', fontsize=fontsize)
        ax.set_xlabel('', fontsize=fontsize)
        ax.set_ylabel('Values', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize * 0.8)
        ax.tick_params(axis='x', rotation=45)

        # 加虚线网格，透明度 0.7
        ax.grid(True, ls='--', alpha=0.7)

    # ---------- 共享 legend ----------
    # 1. 从第一个子图拿 legend 信息
    handles, labels = axes[0].get_legend_handles_labels()
    # 2. 在 figure 级别放 legend，loc 可以调，bbox_to_anchor 控制左右上下
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.535, 0), ncol=2, fontsize=fontsize * 0.8)
    # 3. 去掉每个子图自己的 legend
    for ax in axes:
        ax.get_legend().remove()

    plt.tight_layout(rect=[0.0, 0.1, 1, 1])
    # 顶部给 legend 留点空
    # plt.show()
    plt.savefig(f"./plot_resource/badchain.jpg", dpi=300)
    return


if __name__ == '__main__':
    # plot_DSR()        # 绘制DSR雷达图-已弃用
    # plot_lf_ss()      # 绘制LF和SS分数的散点图
    # plot_badchain()
    # plot_rate()       # 绘制污染率和中毒的关系
    # count_frequency() # 统计数据集里的频率
    plot_frequency()  # 绘制数据集里的频率
    # plot_logits()     # 绘制Logits的主图
    # plot_param()      # 绘制参数敏感性实验
    # plot_ablation()   # 绘制字母z频率和cacc/asr的关系
    # plot_logit_point() # 绘制不同攻击下Logits的散点图