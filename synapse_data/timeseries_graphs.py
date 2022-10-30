import plotly.graph_objects as go
import pickle
import statistics
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

df1_feature = features[500]
df1_label = labels[500]
df2_feature = features[1856]
df2_label = labels[1856]
df3_feature = features[25]
df3_label = labels[25]
df4_feature = features[2498]
df4_label = labels[2498]
features1 = [df1_feature,df1_label,df2_feature,df2_label,df3_feature,df3_label,df4_feature,df4_label]
i = 0
while i < len(features1):
    feature = features1[i]
    label = features1[i+1]
    fig = go.Figure()
    print(feature)
    fig.add_trace(go.Scatter(x=feature['time'], y=feature["heart_rate"], name='heart_rate'))
    fig.add_trace(go.Scatter(x=feature['time'], y=feature["motion"], name='motion'))
    fig.add_trace(go.Scatter(x=feature['time'], y=feature["GSR"], name='GSR'))
    fig.add_trace(go.Scatter(x=[label['time']],text=[label.values.tolist()])) #PANAS_1,PANAS_2,PANAS_3,PANAS_4,PANAS_5,PANAS_6,PANAS_7,PANAS_8,PANAS_9,PANAS_10,Valence,Arousal
    fig.show()
    i += 2
    input()
# fig.update_layout(title=str(df1_label))
# fig.add_trace(go.Scatter(x=df1_feature['time'], y=df1_label["PANAS_2"]))
# fig.add_trace(go.Scatter(x=df1_feature['time'], y=df1_label["PANAS_3"]))
# fig.add_trace(go.Scatter(x=df1_feature['time'], y=df1_label["PANAS_4"]))
# fig.add_trace(go.Scatter(x=df1_feature['time'], y=df1_label["PANAS_5"]))
# fig.add_trace(go.Scatter(x=df1_feature['time'], y=df1_label["PANAS_6"]))
print(statistics.mean(df1_label.values.tolist()))
