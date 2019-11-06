import pandas as pd
import matplotlib.pyplot as plt
import sys

def graph(type):
    filename = type + '/scores.csv'
    df = pd.read_csv(filename)
    
    plt.clf()
    ax = plt.gca()

    df.plot(kind='line', x='num_of_frames', y='avg_score', ax=ax)

    title = 'Average score over frame count for ' + type
    plt.title(title)

    plt.xlabel('Frames')
    plt.ylabel('Score')

    graph = type + '/' + type + '.png'
    plt.savefig(graph)

    print(type + ' graph generated')

### MAIN TRY
if sys.argv[1] == '':
    print('Please enter agent to plot')
else:
    try:
        graph(sys.argv[1])
    except:
        print("Agent not found, did you type the name of it correctly?")