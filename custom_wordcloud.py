import wordcloud as wc

def custom_wordcloud(data, label, drop, color):
    '''
    return an instance of class "wordcloud", with specified data, label and theme
    
    Args:
    data: a pandas dataframe with two columns named "text" and "label"
    label: string, positive or negative, indicating drawing wordclouds for positive or negative tweets
    drop: list, containing words that you don't want to show in the wordcloud, to improve visualization
    color: string, controlling the color theme, with options "spring","summer","autumn" and "winter"
    
    Returns:
    An instance of wordcloud, which can be directly plotted with imshow() in matplotlib.pyplot. 
    '''
    if label=='positive':
        label=4
    else:
        label=0
    words = ' '.join([str(data['text'][i]) for i in range(len(data)) 
                      if data['label'][i]==label])
    for w in drop:
        words = words.replace(w, '')
    figure = wc.WordCloud(font_path='msyh.ttc', width=900, height=900, 
                     background_color="white",colormap=color)
    figure.generate(words)
    return figure