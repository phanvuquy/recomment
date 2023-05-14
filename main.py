
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from joblib import dump
from flask import Flask, jsonify, request

app = Flask(__name__)

# Đọc dữ liệu vào dataframe
df = pd.read_csv("traning.csv")
# chuyển các giá trị từ dạng chuỗi sang dạng số
df = pd.get_dummies(df, columns=['outlook', 'temperature', 'level', 'reputation'])

# tách các thuộc tính và kết quả ra khỏi dataframe
X = df.drop(columns=["id", "play"])
y = df["play"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
#
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

#y_pred = model.predict(X_test)
accuracy = dt.score(X_test, y_test)
print('Accuracy:', accuracy)

# print("Accuracy:", accuracy)
#
#new_data = {'outlook': ['Rain'], 'temperature': ['mild'], 'level': ['normal'], 'reputation': ['high']}

#
# y_pred = dt.predict(new_df)
# print(y_pred)

#dump(df, 'decision_tree_model.joblib')
#model = joblib.load('decision_tree_model.joblib')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    outlook = data['outlook']
    temperature = data['temperature']
    level = data['level']
    reputation = data['reputation']

    new_data = pd.DataFrame({"outlook": outlook, "temperature": temperature, "level": level, "reputation": reputation})

    # chuyển đổi dữ liệu mới sang dataframe
    new_df = pd.DataFrame.from_dict(new_data)

    # áp dụng các bước tiền xử lý dữ liệu
    new_df = pd.get_dummies(new_df, columns=['outlook', 'temperature', 'level', 'reputation'])

    # tìm các thuộc tính bị thiếu so với dữ liệu huấn luyện
    missing_cols = set(X.columns) - set(new_df.columns)

    # thêm các thuộc tính bị thiếu vào dữ liệu mới
    for c in missing_cols:
        new_df[c] = 0

    # đảm bảo cùng thứ tự của các thuộc tính
    new_df = new_df[X.columns]

    prediction = dt.predict(new_df)
    accuracy = dt.score(X_test, y_test)
    result = {'prediction': str(prediction[0]),'accuracy': str(accuracy)}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='localhost',port=5000, debug=True)
