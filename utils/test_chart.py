import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签


def test1():
    types = ["A", "B", "C"]

    actual = [81, 74, 79, 81]
    expected = [100, 69, 70, 100]

    plt.figure(figsize=(9, 8))
    plt.subplot(polar=True)

    theta = np.linspace(0, 2 * np.pi, len(actual))

    plt.plot(theta, actual)
    plt.plot(theta, expected)

    plt.legend(labels=('x1', 'x2', 'x3', 'x4'), bbox_to_anchor=(1, 0.2, 0.2, 0.8), fontsize=15)

    plt.fill(theta, actual, 'b', alpha=0.1)
    plt.fill(theta, expected, 'r', alpha=0.1)

    plt.title('adsda', fontsize=20)
    plt.tick_params(labelsize=20)

    plt.show()


def test2():
    result = {"EDI": 34.04, "CDI": 45.7, "IDI": 50.32, "IDIE": 50.32}
    data_length = len(result)
    angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    labels = list(result.keys())
    labels = np.concatenate((labels, [labels[0]]))
    scores = list(result.values())
    scores = np.concatenate((scores, [scores[0]]))

    fig = plt.figure(figsize=(10, 8), dpi=100)

    ax = plt.subplot(121, polar=True)

    # 绘制3个轴射线
    for i in range(data_length):
        ax.plot([angles[i], angles[i]], [0, 100], '-.', lw=0.5, color='black')

    # 从0至100，以25为步长，绘制坐标刻度围栏
    for i in np.arange(0, 100 + 1, 25):
        ax.plot(angles, (data_length + 1) * [i], '-.', lw=0.5, color='black')

    # 设置坐标轴的最大值和最小值
    ax.set_rlim(0, 100)

    # 设置坐标轴各刻度值，从0至100，以25为步长
    ax.set_rticks(np.arange(0, 100 + 1, 25))

    # 显示得分值
    for a, b in zip(angles, scores):
        ax.text(a, b + 20, '%.00f' % b, ha='center', va='center', fontsize=12, color='b')
        ax.set_thetagrids(angles * 180 / np.pi, labels)

        # 绘制实际得分围线
    ax.fill(angles, scores, color='g')

    # 隐藏最外圈的圆
    ax.spines['polar'].set_visible(False)

    # 隐藏圆形网格线
    ax.grid(False)

    # 极坐标指北
    ax.set_theta_zero_location('N')

    # 设置极径标签显示位置
    ax.set_rlabel_position(0)

    plt.show()


def test3():
    Y = [[0.71, 0.625, 0.93, 0.92, 0.71],
         [0.83, 0.875, 0.93, 0.81, 0.83]]
    labs = ['0', '1', '2', '3', 'xxx']
    N = 4

    theta = np.linspace(0, 360, N, endpoint=False)
    print(theta)  # [  0.  60. 120. 180. 240. 300.]

    # 调整角度使得正中在垂直线上
    adj_angle = theta[-1] + 90 - 360
    theta += adj_angle

    # 将角度转化为单位弧度
    X_ticks = np.radians(theta)  # x轴标签所在的位置
    print(X_ticks)

    # 首尾相连
    X = np.append(X_ticks, X_ticks[0])
    # Y = np.hstack((Y, Y[:, 0].reshape(2, 1)))

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw=dict(projection='polar'))

    # 画图
    ax.plot(X, Y[0], marker='o')
    ax.plot(X, Y[1], marker='o')
    ax.set_xticks(X)

    # 设置背景坐标系
    ax.set_xticklabels(labs, fontsize='large')  # 设置标签
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)  # 将轴隐藏
    ax.grid(axis='y')  # 只有y轴设置grid

    # 设置X轴的grid
    n_grids = np.linspace(0, 1, 6, endpoint=True)  # grid的网格数
    grids = [[i] * (len(X)) for i in n_grids]  # grids的半径
    print(grids)

    for i, grid in enumerate(grids):  # 给grid 填充间隔色
        ax.plot(X, grid, color='grey', linewidth=0.5)
        if (i > 0) & (i % 2 == 0):
            ax.fill_between(X, grids[i], grids[i - 1], color='grey', alpha=0.1)

    plt.show()


def draw():
    ##################################################
    # Load data
    ##################################################
    # y = np.asarray([[0.71, 0.82, 0.625, 0.93, 0.93, 0.92],
    #                 [0.83, 0.89, 0.87, 0.93, 0.875, 1.0]])
    # l = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

    # y = np.asarray([[0.71, 0.625, 0.93, 0.93, 0.92],
    #                 [0.83, 0.875, 0.93, 0.875, 1.0]])
    # l = ['P1', 'P2', 'P3', 'P4', 'P5']

    # y = np.asarray([[0.71, 0.625, 0.93, 0.92],
    #                 [0.83, 0.875, 0.93, 1.0]])
    # l = ['P1', 'P2', 'P3', 'P4']

    y = np.asarray([
        [0.71, 0.93, 0.92],
        # [0.61, 0.67, 0.62],
        # [0.78, 0.34, 0.72],
        # [0.83, 0.93, 1.0]
    ])
    l = ['P1', 'P2', 'P3']

    ##################################################
    # Prepare data
    ##################################################
    theta = np.linspace(0, 2 * np.pi, len(l), endpoint=False)

    if len(l) == 3:
        theta += np.pi / 2
    elif len(l) == 5:
        theta += np.pi / 10

    x = np.append(theta, theta[0])
    y = np.concatenate((y, y[:, 0:1]), axis=1)

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw=dict(projection='polar'))

    for y_ in y:
        ax.plot(x, y_, marker='o')
    for y_ in y:
        ax.fill(x, y_, alpha=0.2)
    ax.set_xticks(theta)

    plt.legend(labels=('x1', 'x2'), fontsize=15)
    ax.set_xticklabels(l, fontsize='large')  # 设置标签
    # ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)  # 将轴隐藏
    ax.grid(axis='y')  # 只有y轴设置grid
    ax.set_rticks(np.arange(0, 1, 0.2))  # 设置坐标轴各刻度值，从0至100，以25为步长

    n_grids = np.linspace(0, 1, 6, endpoint=True)  # grid的网格数
    grids = [[i] * len(x) for i in n_grids]  # grids的半径
    print(grids)

    for i, grid in enumerate(grids):  # 给grid 填充间隔色
        ax.plot(x, grid, color='grey', linewidth=0.5)
        # if (i > 0) & (i % 2 == 0):
        #     ax.fill_between(x, grids[i], grids[i - 1], color='grey', alpha=0.1)

    plt.show()


def draw1():
    ##################################################
    # Load data
    ##################################################
    # y = np.asarray([[0.7145, 0.82, 0.625, 0.93, 0.93, 0.92],
    #                 [0.83, 0.89, 0.87, 0.93, 0.875, 1.0]]) * 100
    # l = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

    y = np.asarray([[0.71, 0.625, 0.93, 0.93, 0.92],
                    [0.83, 0.875, 0.93, 0.875, 1.0]]) * 100
    l = ['P1', 'P2', 'P3', 'P4', 'P5']

    # y = np.asarray([[0.71, 0.625, 0.93, 0.92],
    #                 [0.83, 0.875, 0.93, 1.0]])*100
    # l = ['P1', 'P2', 'P3', 'P4']

    # y = np.asarray([
    #     [0.71, 0.93, 0.8087],
    #     [0.61, 0.67, 0.62],
    #     [0.78, 0.34, 0.72],
    #     [0.83, 0.93, 1.0]
    # ]) * 100
    # l = ['P1', 'P2', 'P3']

    ##################################################
    # Prepare data
    ##################################################
    theta = np.linspace(0, 2 * np.pi, len(l), endpoint=False)
    x = np.append(theta, theta[0])
    y = np.concatenate((y, y[:, 0:1]), axis=1)

    ##################################################
    # Draw data
    ##################################################

    # fig, ax = plt.subplots(figsize=(5, 5),
    #                        subplot_kw=dict(projection='polar'))
    fig = plt.figure(figsize=(5, 5))
    # fig.suptitle("Classification")
    ax = plt.subplot(111, polar=True)

    for i in np.arange(0, 100 + 20, 20):
        ax.plot(x, len(x) * [i], '-', lw=0.5, color='gray')
    for i in range(len(l)):
        ax.plot([x[i], x[i]], [0, 100], '-', lw=0.5, color='gray')

    ax.plot(x, y[0], marker='o')
    ax.fill(x, y[0], alpha=0.2)
    for a, b in zip(x, y[0]):
        ax.text(a, b - 15, b, ha='center', va='center', fontsize=24)  # 设置数值
        ax.text(a, b - 15, b, ha='center', va='center', fontsize=24)  # 设置数值
    ax.spines['polar'].set_visible(False)  # 隐藏最外圈的圆
    ax.grid(False)  # 隐藏圆形网格线
    ax.set_thetagrids(theta * 180 / np.pi, l, fontsize=24)  # 设置标签
    ax.set_theta_zero_location('N')
    ax.set_rlim(0, 100)
    ax.set_rlabel_position(0)
    ax.set_rticks([])
    plt.show()


def draw2():
    # ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿
    c_bg = '#5f97d2'
    c_v = '#b1ce46'
    c_t = '#5f97d2'
    c_i = '#d76364'

    ##################################################
    # Load data
    ##################################################
    # y = np.asarray([[0.7145, 0.82, 0.625, 0.93, 0.93, 0.92],
    #                 [0.83, 0.89, 0.87, 0.93, 0.875, 1.0]]) * 100
    # l = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

    # y = np.asarray([
    #     [0.71, 0.625, 0.93, 0.93, 0.92],
    #     [0.83, 0.875, 0.93, 0.875, 1.0],
    #     [0.71, 0.625, 0.93, 0.93, 0.92],
    #     [0.83, 0.875, 0.93, 0.875, 1.0],
    #     [0.71, 0.625, 0.93, 0.93, 0.92],
    #     [0.83, 0.875, 0.93, 0.875, 1.0],
    # ]) * 100
    # l = [r'$\mathcal{A}$', r'$\mathcal{B}$', r'$\mathcal{E}$', r'$\mathcal{F}$', r'$\mathcal{J}$']

    y = np.asarray([
        [0.71, 0.625, 0.93, 0.92],
        [0.83, 0.875, 0.93, 1.0],
        [0.71, 0.625, 0.93, 0.92],
        [0.83, 0.875, 0.93, 1.0],
        [0.71, 0.625, 0.93, 0.92],
        [0.83, 0.875, 0.93, 1.0],
    ]) * 100
    l = [r'$\mathcal{A}$', r'$\mathcal{B}$', r'$\mathcal{E}$', r'$\mathcal{F}$']
    # l=[]

    # y = np.asarray([
    #     [0.71, 0.93, 0.8087],
    #     [0.78, 0.34, 0.72],
    #     [0.61, 0.67, 0.62],
    #     [0.78, 0.34, 0.72],
    #     [0.61, 0.67, 0.62],
    #     [0.83, 0.93, 1.0]
    # ]) * 100
    # l = [r'$\mathcal{A}$', r'$\mathcal{B}$', r'$\mathcal{C}$']

    ##################################################
    # Prepare data
    ##################################################
    theta = np.linspace(0, 2 * np.pi, len(l), endpoint=False)
    x = np.append(theta, theta[0])
    y = np.concatenate((y, y[:, 0:1]), axis=1)
    print(y)

    ##################################################
    # Draw data
    ##################################################

    fig, axs = plt.subplots(nrows=2,
                            ncols=3,
                            figsize=(15, 10),
                            subplot_kw=dict(projection='polar'))
    axs = axs.flatten()

    # fig = plt.figure(figsize=(5, 5))
    # fig.suptitle("Classification")
    # ax = plt.subplot(111, polar=True)

    for k, ax in enumerate(axs):
        for i in np.arange(0, 100 + 20, 20):
            ax.plot(x, len(x) * [i], '-', lw=0.5, color='gray')
        for i in range(len(l)):
            ax.plot([x[i], x[i]], [0, 100], '-', lw=0.5, color='gray')

        print(y[k])
        ax.plot(x, y[k], marker='o', color=c_bg)
        ax.fill(x, y[k], alpha=0.3, color=c_bg)
        for a, b in zip(x, y[k]):
            ax.text(a, b - 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
        ax.spines['polar'].set_visible(False)  # 隐藏最外圈的圆
        ax.grid(False)  # 隐藏圆形网格线
        ax.set_thetagrids(theta * 180 / np.pi, l, fontsize=36, color='black')  # 设置标签
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 100)
        ax.set_rlabel_position(0)
        ax.set_rticks([])

    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    plt.show()


def draw3():
    # data
    c = np.asarray(['#2284e5', '#3a91e8', '#539feb', '#6caded', '#84bbf0',
                    '#58c566', '#6acc77', '#7dd288', '#8fd999', '#a2dfaa',
                    '#e95e50', '#ee8277', '#f3a69e'])
    x = np.asarray([0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1,
                    2, 2, 2])
    y = np.asarray([96.23, 90.23, 89.34, 87.09, 94.23,
                    94.23, 95.34, 91.23, 89.34, 90.69,
                    98.34, 96.30, 92.39])
    s = np.asarray([0.1, 0.3, 0.5, 0.7, 1.0,
                    0.1, 0.3, 0.5, 0.7, 1.0,
                    0.1, 0.5, 0.9]) * 5000
    m = ['o', 's', 'o', 's', 'o', 's',
         'o', 's', 'o', 's', 'o', 's',
         'o', 's', 'o']
    a = [0.5, 0.6, 0.7, 0.8, 0.9,
         0.5, 0.6, 0.7, 0.8, 0.9,
         0.5, 0.7, 0.9]

    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    plt.figure(figsize=(9, 12),
               dpi=80,
               facecolor="w",
               edgecolor='k')

    # for i in range(3):
    #     print(s[i])
    for i in range(len(x)):
        plt.scatter(x=x[i],
                    y=y[i],
                    c=c[i],
                    # marker=m[i],
                    s=s[i],
                    alpha=0.8,
                    edgecolors='none')
    plt.ylim((80, 100))
    # plt.grid(False)
    plt.margins(0.3, 0.3, tight=True)
    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
    plt.show()


def draw_label():
    # data
    c = np.asarray(['#2284e5', '#3a91e8', '#539feb', '#6caded', '#84bbf0',
                    '#ff8823', '#ff9c47', '#ffaf6c', '#ffc391',
                    '#58c566', '#6acc77', '#7dd288', '#8fd999', '#a2dfaa',
                    '#e95e50', '#ee8277', '#f3a69e'])
    x = np.asarray([0, 0, 0, 0, 0,
                    1, 1, 1, 1,
                    2, 2, 2, 2, 2,
                    3, 3, 3])
    y = np.asarray([50, 40, 30, 18, 4,
                    50, 40, 30, 17,
                    50, 40, 30, 18, 4,
                    50, 40, 26])
    s = np.asarray([0.1, 0.3, 0.5, 0.7, 1.0,
                    0.1, 0.3, 0.5, 0.9,
                    0.1, 0.3, 0.5, 0.7, 1.0,
                    0.1, 0.5, 0.9]) * 5000
    # m = ['o', 's', 'o', 's', 'o', 's',
    #      'o', 's', 'o', 's', 'o', 's',
    #      'o', 's', 'o']
    # a = [0.5, 0.6, 0.7, 0.8, 0.9,
    #      0.5, 0.6, 0.7, 0.8, 0.9,
    #      0.5, 0.7, 0.9]

    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    plt.figure(figsize=(9, 12),
               dpi=80,
               facecolor="w",
               edgecolor='k')

    # for i in range(3):
    #     print(s[i])
    for i in range(len(x)):
        plt.scatter(x=x[i],
                    y=y[i],
                    c=c[i],
                    # marker=m[i],
                    s=s[i],
                    alpha=0.8,
                    edgecolors='none')
    # plt.ylim((80, 100))
    # plt.grid(False)
    plt.margins(0.2, 1, tight=True)
    # plt.tight_layout(pad=5, w_pad=0, h_pad=0)

    plt.show()


def draw_line():
    x = range(10)
    y = np.random.randn(6, 10)

    c = ['#2284e5', '#58c566', '#e95e50', '#ff8c00', '#673ab7', '#ffea00']

    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")
    fig = plt.figure(figsize=(9, 6))

    for i in range(6):
        sns.lineplot(x=x, y=y[i], c=c[i])

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # draw()
    # draw1()
    # draw2()
    # draw3()
    draw_label()
    # draw_line()
