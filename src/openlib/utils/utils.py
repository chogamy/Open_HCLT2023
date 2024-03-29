




def draw_label(self, weights, labels):
    print("TSNE: fitting start...")
    tsne = TSNE(n_components=2, metric='cosine', random_state=0, n_jobs=4)
    embedding = tsne.fit_transform(weights)

    df = pd.DataFrame(embedding, columns=['x', 'y'])  # x, y 값을 DataFrame으로 변환
    df['label'] = labels  # 라벨을 DataFrame에 추가
    df.to_csv('tsne_feature.csv', index=False)


def plot_confusion_matrix(self, y_true, y_pred, labels, normalize=False, title=None):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        xlabel='Predicted label',
        ylabel='True label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    plt.savefig(self.tsne_path + '.pdf')
    plt.close(fig)

