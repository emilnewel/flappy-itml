import pandas as pd
import matplotlib.pyplot as plt

def graph(type):
    filename = type + '/scores.csv'
    df = pd.read_csv(filename)

    plt.clf()
    ax = plt.gca()

    df.plot(kind='line', x='frame_count', y='avg_score', ax=ax)

    title = 'Average score over frame count for ' + type
    plt.title(title)

    plt.xlabel('Frames')
    plt.ylabel('Score')

    graph = type + '/' + type + '.png'
    plt.savefig(graph)

    print(type + ' graph generated')

### MAIN TRY
try:
    graph('opmc')
    graph('q_learning')
except:
    print("Error: something failed")