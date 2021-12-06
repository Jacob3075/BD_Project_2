from matplotlib import pyplot as plt

accuracy_list = []
precision = []
f1_score = []


def plotting():
    '''
    f = open("metrics.txt", "r")
    content = f.read().splitlines()
    for line in content:
        a_p_f1 = line.strip().split()
        accuracy_list.append(float(a_p_f1[0]))
        precision.append(float(a_p_f1[1]))
        f1_score.append(float(a_p_f1[2]))
    '''
    batches = list(range(1, len(accuracy_list) + 1))
    print(batches, accuracy_list)

    plt.bar(batches, accuracy_list)
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Batch')
    plt.show()

    plt.bar(batches, precision)
    plt.xlabel('Batches')
    plt.ylabel('Precision')
    plt.title('precision per Batch')
    plt.show()

    plt.bar(batches, f1_score)
    plt.xlabel('Batches')
    plt.ylabel('f1-score')
    plt.title('f1-score per Batch')
    plt.show()
