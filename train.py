import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据 (注意：去掉 header=None，因为数据第一行是列名)
train_path = r"D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/train.csv"
test_path = r"D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/test.csv"

# 读取训练集和测试集
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 查看前几行，确保读取正确
print(train_df.head())

# --- 2.1 检查缺失值 ---
print("训练集缺失值情况:\n", train_df.isnull().sum())

# 如果有缺失值，我们可以用“中位数”填充数值，用“众数”填充类别
# (这里写一个简单的自动填充逻辑，防止报错)
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
    else:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(test_df[col].median())

# --- 2.2 去除重复值 ---
train_df.drop_duplicates(inplace=True)

# --- 2.3 分离特征(X)和目标(y) ---
# ID 列对预测分数没有帮助，必须去掉，但测试集的ID要留着最后提交用
X = train_df.drop(['id', 'exam_score'], axis=1)
y = train_df['exam_score']

# 测试集我们也先把ID拿出来保存，剩下的作为特征
test_ids = test_df['id']
X_test = test_df.drop(['id'], axis=1)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# 区分数值列和类别列
numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
categorical_features = ['gender', 'course', 'internet_access', 'study_method', 
                        'sleep_quality', 'facility_rating', 'exam_difficulty']

# 定义数值型数据的处理步骤：补充缺失值 -> 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 定义类别型数据的处理步骤：补充缺失值 -> 独热编码(OneHot)
# handle_unknown='ignore' 很重要，防止测试集中出现训练集没见过的类别导致报错
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合起来
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("预处理管道构建完成！")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 划分训练集和验证集 (80% 训练, 20% 验证)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 模型 1: 线性回归 ---
lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_val)

# --- 模型 2: 随机森林 ---
#rf_model = Pipeline(steps=[('preprocessor', preprocessor),
#                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
#rf_model.fit(X_train, y_train)
#y_pred_rf = rf_model.predict(X_val)

# --- 评估结果 ---
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"R2 Score (拟合优度): {r2:.4f}")

evaluate_model("线性回归 (Linear Regression)", y_val, y_pred_lr)
#evaluate_model("随机森林 (Random Forest)", y_val, y_pred_rf)

# 使用效果最好的模型对测试集进行预测
# 注意：不需要再次 fit，直接 predict，因为管道已经记住了处理逻辑
final_predictions = lr_model.predict(X_test)

# 创建提交的 DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': final_predictions
})

# 检查一下格式
print(submission.head())

# 保存为 CSV 文件
save_path = "D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/submission.csv"
submission.to_csv(save_path, index=False)

print(f"恭喜！预测文件已生成并保存至: {save_path}")