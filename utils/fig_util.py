import matplotlib
import matplotlib as mpl

mpl.use('Agg')
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = 'Times New Roman'

c_bg = '#5f97d2'


def draw_radar_chart(y, l, fig_path):
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
        for i, (a, b) in enumerate(zip(x, y[k])):
            if i == 2 and b > 60:
                ax.text(a, b - 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
            else:
                ax.text(a, b + 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
            # if i in [0, 1]:
            #     ax.text(a, b + 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
            # else:
            #     ax.text(a, b - 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
        ax.spines['polar'].set_visible(False)  # 隐藏最外圈的圆
        ax.grid(False)  # 隐藏圆形网格线
        ax.set_thetagrids(theta * 180 / np.pi, l, fontsize=36, color='black')  # 设置标签
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 100)
        ax.set_rlabel_position(0)
        ax.set_rticks([])

    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    # plt.show()
    plt.savefig(fig_path)


def draw_bubble_chart(y, fig_path):
    # data
    c = np.asarray(['#2284e5', '#3a91e8', '#539feb', '#6caded', '#84bbf0',
                    '#ff8823', '#ff9c47', '#ffaf6c', '#ffc391',
                    '#58c566', '#6acc77', '#7dd288', '#8fd999', '#a2dfaa',
                    '#e95e50', '#ee8277', '#f3a69e'])
    x = np.asarray([0, 0, 0, 0, 0,
                    1, 1, 1, 1,
                    2, 2, 2, 2, 2,
                    3, 3, 3])
    # y = np.asarray([96.23, 90.23, 89.34, 87.09, 94.23,
    #                 94.23, 95.34, 91.23, 89.34, 90.69,
    #                 98.34, 96.30, 92.39])
    s = np.asarray([0.1, 0.3, 0.5, 0.7, 1.0,
                    0.1, 0.3, 0.5, 0.9,
                    0.1, 0.3, 0.5, 0.7, 1.0,
                    0.1, 0.5, 0.9]) * 4000

    plt.style.use('seaborn-whitegrid')

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
    plt.grid(False)
    plt.margins(0.5, 0.5, tight=True)
    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
    # plt.show()
    plt.savefig(fig_path)
