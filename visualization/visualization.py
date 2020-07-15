import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors


def elasticnet_performance(results_df, save_option=False, filename=None):

    alpha = results_df['log_alpha'][results_df['MSE_valid'].idxmin()]
    l1 = results_df['l1_ratio'][results_df['MSE_valid'].idxmin()]

    results_df_l1 = results_df[results_df['l1_ratio'] == l1]
    results_df_alpha = results_df[results_df['log_alpha'] == alpha]

    plt.figure(figsize=(12, 6))
    ax1, ax2 = plt.subplot(121), plt.subplot(122)

    ax1.plot(
        results_df_l1['log_alpha'],
        results_df_l1['MSE_train'],
        'r--', marker='x', label='Train')
    ax1.plot(
        results_df_l1['log_alpha'],
        results_df_l1['MSE_valid'],
        'g--', marker='o', label='Valid')
    ax1.set_title(
        r'log($\alpha$)- MSE by 10Fold Cross Validation($\l$= %.1f)' % l1)
    ax1.set_xlabel(r'log($\alpha$)')
    ax1.legend(loc='lower right', frameon=False)
    ax1.grid(True)

    ax2.plot(
        np.arange(1, len(results_df_alpha) + 1),
        results_df_alpha['MSE_train'],
        'r--', marker='x', label='Train')
    ax2.plot(
        np.arange(1, len(results_df_alpha) + 1),
        results_df_alpha['MSE_valid'],
        'g--', marker='o', label='Valid')
    ax2.set_title(
        r'$\l$- MSE by 10Fold Cross Validation(log($\alpha$)= %d)' % alpha)
    ax2.set_xlabel(r'$\l$')
    ax2.legend(loc='lower right', frameon=False)
    ax2.set_xticks(
        np.arange(1, len(results_df_alpha) + 1),
        ['%.3f' % l1 for l1 in results_df_alpha['l1_ratio']])
    ax2.grid(True)

    if save_option:
        plt.savefig('../results/%s' % filename, dpi=300)


def visualize_rating(ratings, save_option=False, filename=None):

    plt.figure(figsize=(8, 6))

    if len(ratings.value_counts()) > 5:
        plt.hist(ratings, bins=50, color='g')
    else:
        y = ratings.astype(int).value_counts().sort_index()
        x = y.index
        plt.bar(np.arange(1, len(x) + 1), y, color='g')
        plt.xticks(np.arange(1, len(x) + 1), x)

    # mean= ratings.mean()

    plt.title(r'Rating Distribution')
    plt.xlabel(r'Rating')
    plt.ylabel(r'Number of Reviews')

    if save_option:
        plt.savefig(r'../results/%s.png' % filename, dpi=1200)

    plt.show()


def visualize_factor_score(
    mean_df, attraction,
    drop_option=False, drop_values=(0, 0),
        save_option=False, filename=None, korean=True):

    from matplotlib import font_manager, rc
    import matplotlib

    font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 16})

    attraction_mean = mean_df[mean_df['spot'] == attraction]
    attraction_mean = attraction_mean.iloc[:-1, :].reset_index(drop=True)
    cols = mean_df.columns[2:]
    date_list = \
        attraction_mean['date'].str[2:4]+'/'+attraction_mean['date'].str[4:]

    if drop_option:
        print_cols = attraction_mean[cols].iloc[-1, :].transpose()
        print_cols = print_cols.sort_values(ascending=False).index

        if drop_values[0] == 0:
            print_cols = print_cols[-drop_values[1]:]
        elif drop_values[1] == 0:
            print_cols = print_cols[:drop_values[0]]
        else:
            print_cols = np.append(
                print_cols[:drop_values[0]], print_cols[-drop_values[1]:])

    else:
        print_cols = cols

    num_colors = len(cols)
    cm = plt.get_cmap('tab20')
    cNorm = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(num_colors)])

    for i in range(num_colors):
        ax.plot(date_list,
                attraction_mean[cols[i]],
                label='%s' % cols[i],
                marker='.',
                linestyle='-')

        ax.set_xticklabels(labels=date_list, fontsize=5)
        ax.tick_params(axis='y', which='major', labelsize=10)

        if cols[i] in print_cols:
            ax.text(
                date_list.unique()[-1],
                attraction_mean[cols[i]].values[-1],
                '%s' % cols[i].replace('_', ' '),
                fontsize=10)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Factor Score', fontsize=12)
    spot = attraction_mean['spot'].unique()[0].replace('_', ' ')
    fig.suptitle('%s Monthly Factor Score' % spot, fontsize=16)

    if save_option:
        plt.savefig(r'../results/plot/%s.png' % filename, dpi=600)

    plt.show()
