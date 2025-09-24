import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from pandas.tests.series.methods.test_rank import results


def plot_redar():
    plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    plt.switch_backend('agg')  # 添加此行代码
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    # Spectral, tab20c_r, Set3_r, PuOr, Paired_r
    # colors = sns.color_palette("Paired", 6)
    fontsize = 20

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

        # labels = df.columns[2:]
        # for i, label in enumerate(labels):
        #     if i == 1:  # Second label
        #         ax.text(angles[i], 0, label, size=fontsize, ha='left', transform=ax.transData)
        #     else:
        #         ax.text(angles[i], 0, label, size=fontsize, ha='center', transform=ax.transData)
        # ax.set_xticklabels([])  # Remove default labels

    all_entities = [r'Wordbkd', r'Sentbdk', r'Stylebkd', r'Synbkd', r'\texttt{BadFreq}']
    fig.legend(all_entities, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(all_entities), fontsize=fontsize)
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

    colors = ['#9467BC', '#007F00', '#2878B4', '#D62728']
    markers = ['p', 's', 'o', '^']
    # d = {0: '(a)', 1: '(b)', 2: '(c)'}

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
        # ax.set_yticks([0.5, 0.6, 0.70, 0.80, 0.90, 1.0])
        # ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])  # 设置刻度标签为实际的rate值

        if i == 0:
            ax.set_yticks([0.70, 0.80, 0.90, 1.0])
            ax.set_xticklabels(group["rate"])  # 设置刻度标签为实际的rate值
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
    fig, axs = plt.subplots(2, 3, figsize=(20, 6))
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
        axs[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[i].set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=fontsize*0.8)

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

        # if attacker == 'BadFreq':
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
    # plt.rcParams['text.usetex'] = True
    # plt.switch_backend('agg')  # 添加此行代码

    # 读取CSV文件
    df = pd.read_csv('./plot_resource/param.csv')

    # 设置字体大小的变量
    fontsize = 18

    # 创建figure和两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    markers = ['o', 'x', '*']
    # 绘制子图1: 随k值变化
    for ii, dataset in enumerate(df['dataset'].unique()):
        # 筛选相关数据
        dataset_data = df[(df['dataset'] == dataset) & (df['p-name'] == 'k')]

        # 如果没有数据则跳过
        if dataset_data.empty:
            continue

        # 绘制均值线
        line, = ax1.plot(dataset_data['param'], dataset_data['mean'],
                         label=dataset, marker=markers[ii], linewidth=2, markersize=8)

        # 绘制std的阴影区域
        ax1.fill_between(dataset_data['param'],
                         dataset_data['mean'] - dataset_data['std'],
                         dataset_data['mean'] + dataset_data['std'],
                         color=line.get_color(), alpha=0.2)

    # 设置子图1的样式
    ax1.set_xlabel(r'$k$', fontsize=fontsize)
    ax1.set_ylabel('US scores', fontsize=fontsize)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_yticklabels(['-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=fontsize*0.8)
    ax1.set_xticks([0, 2, 4, 6, 8])
    ax1.set_xticklabels(['0', '2', '4', '6', '8'], fontsize=fontsize*0.8)

    # 绘制子图2: 随sigma值变化
    for ii, dataset in enumerate(df['dataset'].unique()):
        # 筛选相关数据
        dataset_data = df[(df['dataset'] == dataset) & (df['p-name'] == 'sigma')]

        # 绘制均值线
        line, = ax2.plot(dataset_data['param'], dataset_data['mean'],
                         label=dataset, marker=markers[ii], linewidth=2, markersize=8)

        # 绘制std的阴影区域
        ax2.fill_between(dataset_data['param'],
                         dataset_data['mean'] - dataset_data['std'],
                         dataset_data['mean'] + dataset_data['std'],
                         color=line.get_color(), alpha=0.2)

    # 设置子图2的样式
    ax2.set_xlabel(r'$\epsilon$', fontsize=fontsize)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_yticklabels(['-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=fontsize*0.8)
    ax2.set_xticks([0, 0.007, 0.014, 0.021, 0.028])
    ax2.set_xticklabels(['0.000', '0.007', '0.014', '0.021', '0.028'], fontsize=fontsize*0.8)

    # 创建全局图例并放置在底部中心
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.18),
               ncol=3, fancybox=True, fontsize=fontsize*0.8)

    # 调整整个figure的布局，为图例留出空间
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    plt.savefig("./plot_resource/param.jpg", dpi=300)


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
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    df = pd.read_csv('./plot_resource/lf-ss.csv')

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # 2-a. 先画散点：Task → marker；Attacker → 颜色
    sns.scatterplot(
        data=df,
        x='SS', y='LF',
        hue='attackers',
        style='task',
        s=120,
        palette='tab10',
        edgecolor='k',
        ax=ax
    )

    # 2-b. 为每个 attacker 画覆盖椭圆
    attacker_colors = dict(zip(df['attackers'].unique(),
                               sns.color_palette('tab10', n_colors=df['attackers'].nunique())))

    for atk, color in attacker_colors.items():
        sub = df[df['attackers'] == atk].dropna(subset=['SS'])
        if sub.empty:
            continue
        x = sub['SS'].values
        y = sub['LF'].values
        # 计算椭圆参数
        mean = np.array([x.mean(), y.mean()])
        cov = np.cov(x, y)
        # 计算覆盖所有点的椭圆：半径取最大马氏距离
        delta = np.column_stack([x - mean[0], y - mean[1]])
        mahal = np.sqrt((delta @ np.linalg.inv(cov) * delta).sum(axis=1))
        radius = mahal.max()
        # 椭圆宽高
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        width = 2 * radius * np.sqrt(eigvals[0])
        height = 2 * radius * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        # 创建并添加椭圆
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=color, alpha=0.15, edgecolor='none')
        ax.add_patch(ell)

    # 3. 美化
    plt.xlabel('SS')
    plt.ylabel('LF')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./plot_resource/lf-ss.jpg", dpi=300)
    # plt.show()
    return


def plot_frequency():
    df = pd.read_csv('./plot_resource/freq.csv')
    tasks = ['Classification', 'Jailbreak', 'Generation']
    color = '#0075B8'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ii, ax, task in zip(['(a)', '(b)', '(c)'], axes, tasks):
        mean_col = f'{task}_mean'
        var_col = f'{task}_var'

        bars = ax.bar(df['letter'], df[mean_col], color=color, edgecolor='black')
        ax.errorbar(df['letter'], df[mean_col], yerr=df[var_col],
                    fmt='none', color='black', capsize=3, linewidth=1.2)

        ax.set_title(f'{ii} {task} Task', fontsize=18)
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

if __name__ == '__main__':
    # plot_redar()
    plot_lf_ss()
    # plot_rate()
    # plot_frequency()
    # plot_logits()
    # plot_param()
    # count_frequency()