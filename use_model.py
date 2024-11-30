from catboost import CatBoostClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = CatBoostClassifier()
model.load_model('catboost_model.bin')


def use_model(path_to_data: str):
    df = pd.read_csv(path_to_data)
    df_label = df.copy()
    for col in df.select_dtypes(include='object').columns:
        value_counts = df_label[col].value_counts()
        rare_values = value_counts[value_counts < 2500].index
        df_label[col] = df_label[col].apply(lambda x: 'SMALL' if x in rare_values else x)

        label_encoder = LabelEncoder()
        df_label[col] = label_encoder.fit_transform(df_label[col])

    df['res'] = model.predict(df_label)

    anomaly = df[df['res'] == True]
    anomaly_index = list(anomaly.index)

    res = []
    for i in df.index:
        if i in anomaly_index:
            res.append('True')
        else:
            res.append('False')

    with open("preds_use_model.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(res))

    print("Результат сохранен в -> preds_use_model.csv")

    return True


use_model("normal.csv")
